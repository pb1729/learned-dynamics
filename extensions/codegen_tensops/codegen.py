

PRELUDE_CPP = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
"""

PRELUDE_CU = """
#define WARPSZ 32

"""

FNARG_TAB = "    "


def tab(s):
  if isinstance(s, list):
    return ["  " + line for line in s]
  return "\n".join(tab(s.split("\n")))


def coeffs_to_base_idxs(coeffs):
  ans = [0]
  for coeff in coeffs:
    ans.append(ans[-1] + coeff)
  return ans


def at(indices, shape):
  """ codegen a multidimensional index. the first elem of shape is unused and can be ommitted
      so len(shape) == len(indices) - 1 is allowed along with the case of equal lengths. """
  *indices_rest, index_last = indices
  if len(indices_rest) == 0:
    return f"[{index_last}]"
  *shape_rest, dim_last = shape
  return f"[({at(indices_rest, shape_rest)[1:-1]})*{dim_last} + {index_last}]"

def ilabel(*inds):
  return "_" + "".join([str(i) for i in inds])

class LocalRef:
  def __init__(self, varnm):
    self.varnm = varnm
  def tensidx(self, i, tsz):
    return f"{self.varnm}{ilabel(i)}"

class IndexRef:
  def __init__(self, ptrnm, base_indices, base_shape):
    self.ptrnm = ptrnm
    self.base_indices = base_indices
    self.base_shape = base_shape
  def tensidx(self, i, tsz):
    return f"{self.ptrnm}{at(list(self.base_indices) + [i,], list(self.base_shape) + [tsz,])}"


class SharedVars:
  """ This class keeps track of variables in shared memory for us.
      Variables are grouped into 'classes' that each have a particular chunksz.
      The chunksz will be a variable with some name. You can define this variable however you like.
      Variables in such a class have a size that is a compile-time constant times the chunksz. """
  def __init__(self):
    # note: this code uses python's consistent dict ordering feature
    self.coeffs = {}    # chunksz: [coeff, ...]
    self.variables = {} # varnm:   (chunksz, idx)
  def add_variable(self, varnm, chunksz, coeff):
    assert isinstance(coeff, int)
    if chunksz not in self.coeffs:
      self.coeffs[chunksz] = []
    self.variables[varnm] = (chunksz, len(self.coeffs[chunksz]))
    self.coeffs[chunksz].append(coeff)
  def sharedmemsz_calculations(self):
    ans = ["int sharedmemsz = 0;"]
    for i, chunksz in enumerate(self.coeffs):
      if i > 0: # first offset is a freebie: it's always 0
        ans.append(f"int {chunksz}_base = sharedmemsz;")
      ans.append(f"sharedmemsz += {sum(self.coeffs[chunksz])}*{chunksz};")
    return "\n".join(ans)
  def _chunksz_calculations(self, chunksz_exprs):
    return "\n".join([
      f"int {chunksz} = {chunksz_exprs[chunksz]};"
      for chunksz in self.coeffs
      if chunksz_exprs[chunksz] != chunksz # this feature allows us to reuse identical variables!
    ])
  def params(self):
    ans = []
    for i, chunksz in enumerate(self.coeffs):
      if i > 0: # first offset is a freebie: it's always 0
        ans.append(f"int {chunksz}_base")
      ans.append(f"int {chunksz}")
    return ", ".join(ans)
  def args(self):
    ans = []
    for i, chunksz in enumerate(self.coeffs):
      if i > 0: # first offset is a freebie: it's always 0
        ans.append(f"{chunksz}_base")
      ans.append(f"{chunksz}")
    return ", ".join(ans)
  def vars_readout(self):
    base_idxs = {
      chunksz: coeffs_to_base_idxs(self.coeffs[chunksz])
      for chunksz in self.coeffs
    }
    ans = ["extern __shared__ float s[];"]
    for chunksz in self.coeffs:
      first_chunksz = chunksz
      break
    for var in self.variables:
      chunksz, idx = self.variables[var]
      coeff = self.coeffs[chunksz][idx]
      base_idx = base_idxs[chunksz][idx]
      base_expr = "" if chunksz == first_chunksz else f"{chunksz}_base + "
      ans.append(f"float* {var} = &s[{base_expr}{base_idx}*{chunksz}]; // size = {coeff}*{chunksz}")
    return "\n".join(ans)




class Function:
  """ Defines a C++ function that wraps a CUDA function. Can accept integer and tensor float parameters.
      Function return type is void but it can write to output tensors.
      Given some information about the function, such as a SharedVars() instance, and a function body using
      the arguments and the shared vars, this class provides methods to declare and predeclare the C++ wrapper,
      and to declare the cuda kernel. """
  def __init__(self, fnname, int_params, tens_params, tens_outputs, shapes,
      gridsz_expr, blocksz_expr, shared_vars, chunksz_exprs, cu_body):
    self.fnname = fnname
    self.int_params = int_params
    self.tens_params = tens_params
    self.tens_outputs = tens_outputs
    self.shapes = shapes
    self.gridsz_expr = gridsz_expr
    self.blocksz_expr = blocksz_expr
    self.shared_vars = shared_vars
    self.chunksz_exprs = chunksz_exprs
    self.cu_body = cu_body
    self.do_checks = True
  def _get_kern_int_params(self):
    # int params are slightly different for the kernel itself...
    return [
      int_param
      for int_param in self.int_params
      if int_param not in self.shared_vars.coeffs # prevent duplication of identical variables!
    ]
  def _stub_cpp(self):
    return "".join([
      f"std::vector<at::Tensor> {self.fnname}_cuda(\n",
      FNARG_TAB, ", ".join([f"const at::Tensor& {tens_param}" for tens_param in self.tens_params]), ")"
    ])
  def _stub(self):
    return "".join([
      f"void {self.fnname}(\n",
      FNARG_TAB, ", ".join([f"int {int_param}" for int_param in self.int_params]), ",\n",
      FNARG_TAB, ", ".join([f"const float* {tens_param}" for tens_param in self.tens_params]), ",\n",
      FNARG_TAB, ", ".join([f"float* {tens_output}" for tens_output in self.tens_outputs]), ")"
    ])
  def _stub_kern(self):
    return "".join([
      "__global__\n",
      f"void {self.fnname}_kern(\n",
      FNARG_TAB, f"// <<<({self.gridsz_expr}), ({self.blocksz_expr})>>>\n",
      FNARG_TAB, self.shared_vars.params(), ",\n",
      FNARG_TAB, ", ".join([f"int {int_param}" for int_param in self._get_kern_int_params()]), ",\n",
      FNARG_TAB, ", ".join([f"const float* {tens_param}" for tens_param in self.tens_params]), ",\n",
      FNARG_TAB, ", ".join([f"float* __restrict__ {tens_output}" for tens_output in self.tens_outputs]), ")"
    ])
  def _call_kern(self, gridsz, blocksz, sharedsz):
    return "".join([
      f"{self.fnname}_kern<<<{gridsz}, {blocksz}, {sharedsz}>>>(\n",
      FNARG_TAB, self.shared_vars.args(), ",\n",
      FNARG_TAB, ", ".join([f"{int_param}" for int_param in self._get_kern_int_params()]), ",\n",
      FNARG_TAB, ", ".join([f"{tens_param}" for tens_param in self.tens_params]), ",\n",
      FNARG_TAB, ", ".join([f"{tens_output}" for tens_output in self.tens_outputs]), ")"
    ])
  def _call(self):
    return "".join([
      f"{self.fnname}(\n",
      FNARG_TAB, ", ".join([f"{int_param}" for int_param in self.int_params]), ",\n",
      FNARG_TAB, ", ".join([f"reinterpret_cast<float*>({tens_param}.data_ptr<float>())" for tens_param in self.tens_params]), ",\n",
      FNARG_TAB, ", ".join([f"reinterpret_cast<float*>({tens_output}.data_ptr<float>())" for tens_output in self.tens_outputs]), ");"
    ])
  def _checks(self):
    ans = []
    initialized_dims = set()
    if self.do_checks:
      for tens_param in self.tens_params:
        ans.append(f"CHECK_INPUT({tens_param});")
    for tens_param in self.tens_params:
      ans.append(f"at::Device device = {tens_param}.device();")
      ans.append("cudaSetDevice(device.index()); // run kernel on same device as input tensors")
      break
    for tens_param in self.tens_params:
      if self.do_checks:
        ans.append(f"TORCH_CHECK({tens_param}.dim() == {len(self.shapes[tens_param])}, \"{tens_param} has wrong number of axes\");")
      for i, dim in enumerate(self.shapes[tens_param]):
        if isinstance(dim, int) or dim in initialized_dims:
          if self.do_checks:
            ans.append(f"TORCH_CHECK({tens_param}.size({i}) == {dim}, \"{tens_param}: expected axis {i} to have size {dim}\");")
        else:
          ans.append(f"int {dim} = {tens_param}.size({i});")
          initialized_dims.add(dim)
    for int_param in self.int_params:
      assert int_param in initialized_dims, f"int param {int_param} was never initialized"
    return ans
  def _output_allocs(self):
    ans = []
    for tens_output in self.tens_outputs:
      shape = "{" + ", ".join([str(dim) for dim in self.shapes[tens_output]]) + "}"
      ans.append(f"at::Tensor {tens_output} = torch::empty({shape}, torch::dtype(torch::kFloat32).device(device));")
    return ans
  def predeclare(self):
    return self._stub() + ";"
  def define_cpp(self):
    return_value = "{" + ", ".join(self.tens_outputs) + "}"
    return "\n".join([
      self._stub_cpp() + " {",
      *tab(self._checks()),
      *tab(self._output_allocs()),
      tab(self._call()),
      f"  return {return_value};",
      "}"])
  def define(self):
    kern_call = self._call_kern("gridsz", "blocksz", "sharedmemsz*sizeof(float)")
    return (self._stub() + " {\n" + tab(f"""
{self.shared_vars._chunksz_calculations(self.chunksz_exprs)}
{self.shared_vars.sharedmemsz_calculations()}
dim3 gridsz = dim3({self.gridsz_expr});
dim3 blocksz = dim3({self.blocksz_expr});
{kern_call};
"""
    ) + "\n}\n")
  def define_kern(self):
    return self._stub_kern() + " {\n" + tab(self.shared_vars.vars_readout()) + "\n" + tab(self.cu_body) + "\n}\n"
  def get_binding(self):
    return f"m.def(\"{self.fnname}_cuda\", &{self.fnname}_cuda, \"{self.fnname}_cuda({', '.join(self.tens_params)})\");"


def warp_sum(variables):
  """ after this, for each variable, the first thread in the warp will have the sum. """
  return [
    "// reduce across the warp so that first thread in warp will have the sum ",
    "for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {",
    *[
      f"  {var} += __shfl_down_sync(0xffffffff, {var}, offset);"
      for var in variables
    ],
    "}"
  ]


def prodlabel(prod):
  return f"_{prod[0]}{prod[1]}{prod[2]}"



def compute_left(tens_sizes, prod):
  inds_l, inds_r, inds_o = prod
  label = prodlabel(prod)
  ans = []
  for i in range(tens_sizes[inds_l]):
    ans.append(f"float accum{label}{ilabel(i)} = 0.0;")
  ans.append(f"for (int idx_chan_in{label} = threadIdx.x; idx_chan_in{label} < dim_{inds_l}; idx_chan_in{label} += blockDim.x) {{")
  W_index = at(["threadIdx.y", f"idx_chan_in{label}"], ["dim_l", f"dim_{inds_l}"])
  ans.append(f"  float W_oi{label} = W{label}{W_index};")
  for i in range(tens_sizes[inds_l]):
    x_index = at(["idx_batch", f"idx_chan_in{label}", str(i)], ["batch", f"dim_{inds_l}", str(tens_sizes[inds_l])])
    ans.append(f"  accum{label}{ilabel(i)} += W_oi{label}*x_{inds_l}{x_index};")
  ans.append("}")
  ans.extend(warp_sum([f"accum{label}{ilabel(i)}" for i in range(tens_sizes[inds_l])]))
  ans.append("if (threadIdx.x == 0) {")
  for i in range(tens_sizes[inds_l]):
    ans.append(f"  left{label}{at(['threadIdx.y', str(i)], ['dim_l', str(tens_sizes[inds_l])])} = accum{label}{ilabel(i)};")
  ans.append("}")
  return ans

def tensprod(tens_sizes, prod, leftref, rightref, outref):
  inds_l, inds_r, inds_o = prod
  label = prodlabel(prod)
  double_n_reduces = (inds_l + inds_r - inds_o)
  assert double_n_reduces % 2 == 0, f"tensor product {inds_l}, {inds_r} -> {inds_o} has incorrect parity"
  n_reduces = double_n_reduces//2
  assert 0 <= n_reduces <= min(inds_l, inds_r), f"tensor product {inds_l}, {inds_r} -> {inds_o} is impossible"
  ans = []
  for i in range(tens_sizes[inds_o]):
    tensprod = []
    tsz_reduce = tens_sizes[n_reduces]
    for j in range(tsz_reduce):
      free_tsz_right = tens_sizes[inds_r]//tsz_reduce
      i_left = i//free_tsz_right
      i_right = i % free_tsz_right
      tensprod.append(
        leftref.tensidx(i_left*tsz_reduce + j, tens_sizes[inds_l])
        + "*" +
        rightref.tensidx(i_right*tsz_reduce + j, tens_sizes[inds_r]))
    tensprod = " + ".join(tensprod)
    ans.append(f"{outref.tensidx(i, tens_sizes[inds_o])} = ({tensprod});")
  return ans

def tensor_prods(name, max_inds, prods, dim_l):
  """ codegen a fused kernel that does many kinds of tensor products at once """
  V = SharedVars()
  tens_sizes = [3**p for p in range(max_inds + 1)]
  def indsgen():
    return range(len(tens_sizes))
  int_params = ["batch", *[f"dim_{i}" for i in indsgen()]]
  tens_params = [f"x_{i}" for i in indsgen()]
  tens_outputs = [f"y_{i}" for i in indsgen()]
  chunksz_exprs = {"dim_l": f"{dim_l}"}
  chunksz_exprs.update({
    f"dim_{i}": f"dim_{i}"
    for i in indsgen()
  })
  chunksz_exprs.update({
    f"p_{i}": f"dim_l*dim_{i}"
    for i in indsgen()
  })
  gridsz_expr = f"batch"
  blocksz_expr = f"WARPSZ, dim_l"
  # shapes setup
  shapes = {}
  for i in indsgen():
    shapes[f"x_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes[f"y_{i}"] = ("batch", f"dim_{i}") + (3,)*i
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    shapes[f"W{label}"] = (dim_l, f"dim_{inds_l}")
    shapes[f"P{label}"] = (f"dim_{inds_o}", dim_l, f"dim_{inds_r}")
  # Now for the hard part: generating the main computational part of the kernel!
  cu_body = []
  # COMPUTE LEFT SIDE OF PRODUCT:
  cu_body.append("{ // linear transform to compute the left sides of the products")
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    tens_params.append(f"W{label}")
    V.add_variable(f"left{label}", "dim_l", tens_sizes[inds_l])
    # matmul:
    cu_body.extend(tab(compute_left(tens_sizes, prod)))
  cu_body.append("}")
  cu_body.append("__syncthreads();")
  # TENSOR PRODUCTS:
  cu_body.append("{ // compute tensor products")
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    V.add_variable(f"product{label}", f"p_{inds_r}", tens_sizes[inds_o])
    for i in range(tens_sizes[inds_l]):
      left_index = at(["threadIdx.y", str(i)], ["dim_l", str(tens_sizes[inds_l])])
      cu_body.append(f"  float l{label}{ilabel(i)} = left{label}{left_index};")
    cu_body.append(f"  for (int idx_chan_in{label} = threadIdx.x; idx_chan_in{label} < dim_{inds_r}; idx_chan_in{label} += blockDim.x) {{")
    cu_body.extend(tab(tab(
      tensprod(tens_sizes, prod,
        LocalRef(f"l{label}"),
        IndexRef(f"x_{inds_r}", ["idx_batch", f"idx_chan_in{label}"], ["batch", f"dim_{inds_r}"]),
        IndexRef(f"product{label}", [f"threadIdx.y", f"idx_chan_in{label}"], ["dim_l", f"dim_{inds_r}"]))
    )))
    cu_body.append("  }")
  cu_body.append("}")
  cu_body.append("__syncthreads();")
  # FINAL LINEAR TRANSFORMS:
  cu_body.append("{ // linear transforms to compute the outputs")
  for inds_o in indsgen(): # by inds_o so that we can sum contributions from each prod
    cu_body.append(f"  for (int idx_chan_out_{inds_o} = threadIdx.y; idx_chan_out_{inds_o} < dim_{inds_o}; idx_chan_out_{inds_o} += blockDim.y) {{")
    for i in range(tens_sizes[inds_o]):
      cu_body.append(f"    float y_o_{inds_o}{ilabel(i)} = 0.0;")
    for prod in prods:
      inds_l, inds_r, actual_inds_o = prod
      if inds_o == actual_inds_o:
        label = prodlabel(prod)
        tens_params.append(f"P{label}")
        # matmul
        for i in range(tens_sizes[inds_o]):
          cu_body.append(f"    float accum{label}{ilabel(i)} = 0.0;")
        cu_body.append(f"    for (int idx_chan_in{label} = threadIdx.x; idx_chan_in{label} < dim_l*dim_{inds_r}; idx_chan_in{label} += blockDim.x) {{")
        P_index = at([f"idx_chan_out_{inds_o}", f"idx_chan_in{label}"], [f"dim_{inds_o}", f"dim_l*dim_{inds_r}"])
        cu_body.append(f"      float P_oi{label} = P{label}{P_index};")
        for i in range(tens_sizes[inds_o]):
          product_index = at([f"idx_chan_in{label}", str(i)], [f"dim_{inds_o}", str(tens_sizes[inds_o])])
          cu_body.append(f"      accum{label}{ilabel(i)} += P_oi{label}*product{label}{product_index};")
        cu_body.append("    }")
        cu_body.extend(tab(tab(warp_sum([f"accum{label}{ilabel(i)}" for i in range(tens_sizes[inds_o])]))))
        cu_body.append("    if (threadIdx.x == 0) {")
        for i in range(tens_sizes[inds_o]):
          cu_body.append(f"      y_o_{inds_o}{ilabel(i)} += accum{label}{ilabel(i)};")
        cu_body.append("    }")
    cu_body.append("    if (threadIdx.x == 0) {")
    for i in range(tens_sizes[inds_o]):
      y_index = at([f"idx_batch", f"idx_chan_out_{inds_o}", i], ["batch", f"dim_{inds_o}", tens_sizes[inds_o]])
      cu_body.append(f"      y_{inds_o}{y_index} = y_o_{inds_o}{ilabel(i)};")
    cu_body.append("    }")
    cu_body.append("  }")
  cu_body.append("}")
  # wrap the whole thing in a for loop to make this work with large batches
  cu_body = "\n".join([
    "for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {",
    *tab(cu_body), "}"
  ])
  return Function(name, int_params, tens_params, tens_outputs, shapes, gridsz_expr, blocksz_expr, V, chunksz_exprs, cu_body)


def bindings(functions):
  return "\n".join([
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
    *tab([
      function.get_binding()
      for function in functions
    ]),
    "}"])



if __name__ == "__main__":
  print("TESTING CODEGEN\n")
  if False:
    print("\ntest SharedVars")
    V = SharedVars()
    V.add_variable("v0", "ankh", 1)
    V.add_variable("u0", "morpork", 1)
    V.add_variable("v1", "ankh", 3)
    V.add_variable("u1", "morpork", 3)
    V.add_variable("v2", "ankh", 9)
    V.add_variable("u2", "morpork", 9)
    print(V.params())
    print(V.args())
    print()
    print(V.vars_readout())
    print()
    print(V.sharedmemsz_calculations())
    print()
  if True:
    F = tensor_prods("fused_tensor_prods_example", 1, [(0, 0, 0), (0, 1, 1), (1, 1, 0)], 8)
    print(F.define_kern())
    print()
    print(F.define())
    print()
    print(F.define_cpp())
    print()
    print(bindings([F]))
