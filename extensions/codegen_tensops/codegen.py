from typing_extensions import List, Tuple
from itertools import product


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
#define MODWARP(X) (X & 0x1f)

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

def flat_tensidx(indices):
  """ like at, but specifically for tensors (all dimensions are 3), and with static integer indices """
  if len(indices) == 0: return 0
  *indices_rest, index_last = indices
  return index_last + 3*flat_tensidx(indices_rest)

def tensidx_iter(inds):
  return product(*[range(3) for i in range(inds)])


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
        ans.append(f"int {chunksz}_base, ")
      ans.append(f"int {chunksz}, ")
    return "".join(ans)
  def args(self):
    ans = []
    for i, chunksz in enumerate(self.coeffs):
      if i > 0: # first offset is a freebie: it's always 0
        ans.append(f"{chunksz}_base, ")
      ans.append(f"{chunksz}, ")
    return "".join(ans)
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
      FNARG_TAB, self.shared_vars.params(), "\n",
      FNARG_TAB, ", ".join([f"int {int_param}" for int_param in self._get_kern_int_params()]), ",\n",
      FNARG_TAB, ", ".join([f"const float* {tens_param}" for tens_param in self.tens_params]), ",\n",
      FNARG_TAB, ", ".join([f"float* __restrict__ {tens_output}" for tens_output in self.tens_outputs]), ")"
    ])
  def _call_kern(self, gridsz, blocksz, sharedsz):
    return "".join([
      f"{self.fnname}_kern<<<{gridsz}, {blocksz}, {sharedsz}>>>(\n",
      FNARG_TAB, self.shared_vars.args(), "\n",
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


def sign(perm):
  """ computes the sign of a permutation """
  ans = 1
  for i in range(len(perm)):
    for j in range(i):
      if perm[i] == perm[j]: return 0
      if perm[i] < perm[j]: ans *= -1
  return ans
def gen_tensprod(tens_sizes, prod, leftref, rightref, outref, result=2, chiral=None):
  """ codegen for an individual tensor product. result (0, 1, or 2) shows which thing is our output.
      (indexes into [left, right, out].) if result is None, then we contract all 3 inputs into a scalar.
      If chiral is True, we contract 2 indices with the Levi-Civita tensor to produce 1, indead of
      contracting 2 indices with each other to produce 0, as ususal. We only support having exactly
      one such "chiral contraction" for now. If chiral is None, we guess what it should be. """
  if chiral is None: # guess whether it's chiral based on parity of prod
    return gen_tensprod(tens_sizes, prod, leftref, rightref, outref, result=result, chiral=(sum(prod) % 2 == 1))
  # actual implementation:
  chirals = 1 if chiral else 0
  inds_l, inds_r, inds_o = prod
  label = prodlabel(prod)
  double_n_reduces = (inds_l + inds_r + chirals - inds_o)
  assert double_n_reduces % 2 == 0, f"tensor product {inds_l}, {inds_r}{', 3' if chiral else ''} -> {inds_o} has incorrect parity"
  n_reduces = double_n_reduces//2
  assert 0 <= n_reduces <= min(inds_l, inds_r), f"tensor product {inds_l}, {inds_r}{', 3' if chiral else ''} -> {inds_o} is impossible"
  if chiral:
    assert 1 == chirals <= n_reduces, f"tensor product {inds_l}, {inds_r}, 3 -> {inds_o} has insufficient reduces for its number of levi-civita tensors"
  ref_triplets = {}
  coeffs = {}
  left_keeps = inds_l - n_reduces
  for i in tensidx_iter(inds_o - chirals):
    for j in tensidx_iter(n_reduces - chirals):
      for k_o in tensidx_iter(chirals):
        for k_l in tensidx_iter(chirals):
          for k_r in tensidx_iter(chirals):
            i_left = i[:left_keeps] + j + k_l
            i_right = i[left_keeps:] + j + k_r
            i_out = i + k_o
            tup = flat_tensidx(i_left), flat_tensidx(i_right), flat_tensidx(i_out)
            ref_triplets[tup] = [
                leftref.tensidx(tup[0], tens_sizes[inds_l]),
                rightref.tensidx(tup[1], tens_sizes[inds_r]),
                outref.tensidx(tup[2], tens_sizes[inds_o]),
              ]
            if chiral:
              coeffs[tup] = sign(k_l + k_r + k_o)
            else:
              coeffs[tup] = 1
  if result is None:
    ans = " + ".join([
      (
        "*".join(ref_triplets[tup])
        if coeffs[tup] == 1 else
        f"({coeffs[tup]})*" + "*".join(ref_triplets[tup])
      )
      for tup in ref_triplets
      if coeffs[tup] != 0
    ])
    return f"({ans})"
  else:
    ans = []
    for i in range(tens_sizes[[inds_l, inds_r, inds_o][result]]):
      assignment_var = None
      assignment_val = []
      for tup in ref_triplets:
        if tup[result] == i and coeffs[tup] != 0:
          ref_triplet = ref_triplets[tup]
          prod = "*".join(ref_triplet[:result] + ref_triplet[result + 1:])
          if coeffs[tup] != 1:
            prod = f"({coeffs[tup]})*" + prod
          assignment_val.append(prod)
          if assignment_var is None:
            assignment_var = ref_triplet[result]
          else:
            assert ref_triplet[result] == assignment_var
      assert assignment_var is not None
      ans.append(assignment_var + " = " + " + ".join(assignment_val) + ";")
  return ans


def batchfor_wrap(code):
  """ Wrap code in the proper for loop if kernel has blockIdx to indicate batch index. """
  return "\n".join([
    "for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {",
    *tab(code), "}"
  ])

def gen_fwd_tensor_prods(input_indsset: List[int], output_indsset: List[int], prods: List[Tuple[int, int, int]], tens_sizes):
  code = []
  V = SharedVars()
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    V.add_variable(f"product{label}", f"p_{inds_r}", tens_sizes[inds_o])
  # COMPUTE TENSOR PRODUCTS:
  code.append("{ // compute tensor products")
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    for i in range(tens_sizes[inds_l]):
      left_index = at(["idx_batch", "threadIdx.y", str(i)], ["batch", "dim_l", str(tens_sizes[inds_l])])
      code.append(f"  float l{label}{ilabel(i)} = left{label}{left_index};")
    code.append(f"  for (int idx_chan_in{label} = threadIdx.x; idx_chan_in{label} < dim_{inds_r}; idx_chan_in{label} += blockDim.x) {{")
    code.extend(tab(tab(
      gen_tensprod(tens_sizes, prod,
        LocalRef(f"l{label}"),
        IndexRef(f"x_{inds_r}", ["idx_batch", f"idx_chan_in{label}"], ["batch", f"dim_{inds_r}"]),
        IndexRef(f"product{label}", [f"threadIdx.y", f"idx_chan_in{label}"], ["dim_l", f"dim_{inds_r}"]))
    )))
    code.append("  }")
  code.append("}")
  code.append("__syncthreads();")
  # FINAL LINEAR TRANSFORMS (FORWARD):
  code.append("{ // linear transforms to compute the outputs")
  for inds_o in output_indsset: # by inds_o so that we can sum contributions from each prod
    code.append(f"  for (int idx_chan_out_{inds_o} = threadIdx.y; idx_chan_out_{inds_o} < dim_{inds_o}; idx_chan_out_{inds_o} += blockDim.y) {{")
    for i in range(tens_sizes[inds_o]):
      code.append(f"    float y_o_{inds_o}{ilabel(i)} = 0.0;")
    for prod in prods:
      inds_l, inds_r, actual_inds_o = prod
      if inds_o == actual_inds_o:
        label = prodlabel(prod)
        # matmul
        for i in range(tens_sizes[inds_o]):
          code.append(f"    float accum{label}{ilabel(i)} = 0.0;")
        code.append(f"    for (int idx_chan_in{label} = threadIdx.x; idx_chan_in{label} < dim_l*dim_{inds_r}; idx_chan_in{label} += blockDim.x) {{")
        P_index = at([f"idx_chan_out_{inds_o}", f"idx_chan_in{label}"], [f"dim_{inds_o}", f"dim_l*dim_{inds_r}"])
        code.append(f"      float P_oi{label} = P{label}{P_index};")
        for i in range(tens_sizes[inds_o]):
          product_index = at([f"idx_chan_in{label}", str(i)], [f"dim_l*dim_{inds_o}", str(tens_sizes[inds_o])])
          code.append(f"      accum{label}{ilabel(i)} += P_oi{label}*product{label}{product_index};")
        code.append("    }")
        code.extend(tab(tab(warp_sum([f"accum{label}{ilabel(i)}" for i in range(tens_sizes[inds_o])]))))
        code.append("    if (threadIdx.x == 0) {")
        for i in range(tens_sizes[inds_o]):
          code.append(f"      y_o_{inds_o}{ilabel(i)} += accum{label}{ilabel(i)};")
        code.append("    }")
    code.append("    if (threadIdx.x == 0) {")
    for i in range(tens_sizes[inds_o]):
      y_index = at([f"idx_batch", f"idx_chan_out_{inds_o}", i], ["batch", f"dim_{inds_o}", tens_sizes[inds_o]])
      code.append(f"      y_{inds_o}{y_index} = y_o_{inds_o}{ilabel(i)};")
    code.append("    }")
    code.append("  }")
  code.append("}")
  return V, batchfor_wrap(code)


def gen_bwd_tensor_prods(input_indsset: List[int], output_indsset: List[int], prods: List[Tuple[int, int, int]], tens_sizes):
  code = []
  V = SharedVars()
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    V.add_variable(f"dproduct{label}", f"p_{inds_o}", tens_sizes[inds_r])
  # COMPUTE TENSOR PRODUCTS:
  code.append("{ // compute tensor products")
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    for i in range(tens_sizes[inds_l]):
      left_index = at(["idx_batch", "threadIdx.y", str(i)], ["batch", "dim_l", str(tens_sizes[inds_l])])
      code.append(f"  float l{label}{ilabel(i)} = left{label}{left_index};")
    code.append(f"  for (int idx_chan_out{label} = threadIdx.x; idx_chan_out{label} < dim_{inds_o}; idx_chan_out{label} += blockDim.x) {{")
    code.extend(tab(tab(
      gen_tensprod(tens_sizes, prod,
        LocalRef(f"l{label}"),
        IndexRef(f"dproduct{label}", [f"threadIdx.y", f"idx_chan_out{label}"], ["dim_l", f"dim_{inds_o}"]),
        IndexRef(f"dy_{inds_o}", ["idx_batch", f"idx_chan_out{label}"], ["batch", f"dim_{inds_o}"]),
        result=1)
    )))
    code.append("  }")
  code.append("}")
  code.append("__syncthreads();")
  # FINAL LINEAR TRANSFORMS (BACKWARDS):
  code.append("{ // linear transforms to compute dx")
  for inds_r in input_indsset: # by inds_r so that we can sum contributions from each prod
    code.append(f"  for (int idx_chan_in_{inds_r} = threadIdx.y; idx_chan_in_{inds_r} < dim_{inds_r}; idx_chan_in_{inds_r} += blockDim.y) {{")
    for i in range(tens_sizes[inds_r]):
      code.append(f"    float dx_o_{inds_r}{ilabel(i)} = 0.0;")
    for prod in prods:
      inds_l, actual_inds_r, inds_o = prod
      if inds_r == actual_inds_r:
        label = prodlabel(prod)
        # matmul
        for i in range(tens_sizes[inds_r]):
          code.append(f"    float accum{label}{ilabel(i)} = 0.0;")
        code.append(f"    for (int idx_l{label} = 0; idx_l{label} < dim_l; idx_l{label} += 1) {{")
        code.append(f"      for (int idx_chan_out{label} = threadIdx.x; idx_chan_out{label} < dim_{inds_o}; idx_chan_out{label} += blockDim.x) {{")
        P_index = at([f"idx_chan_out{label}", f"idx_l{label}", f"idx_chan_in_{inds_r}"], [f"dim_{inds_o}", "dim_l", f"dim_{inds_r}"])
        code.append(f"        float P_oi{label} = P{label}{P_index};")
        for i in range(tens_sizes[inds_r]):
          dproduct_index = at([f"idx_l{label}", f"idx_chan_out{label}", str(i)], ["dim_l", f"dim_{inds_o}", str(tens_sizes[inds_r])])
          code.append(f"        accum{label}{ilabel(i)} += P_oi{label}*dproduct{label}{dproduct_index};")
        code.append("      }")
        code.append("    }")
        code.extend(tab(tab(warp_sum([f"accum{label}{ilabel(i)}" for i in range(tens_sizes[inds_r])]))))
        code.append("    if (threadIdx.x == 0) {")
        for i in range(tens_sizes[inds_r]):
          code.append(f"      dx_o_{inds_r}{ilabel(i)} += accum{label}{ilabel(i)};")
        code.append("    }")
    code.append("    if (threadIdx.x == 0) {")
    for i in range(tens_sizes[inds_r]):
      dx_index = at([f"idx_batch", f"idx_chan_in_{inds_r}", i], ["batch", f"dim_{inds_r}", tens_sizes[inds_r]])
      code.append(f"      dx_{inds_r}{dx_index} = dx_o_{inds_r}{ilabel(i)};")
    code.append("    }")
    code.append("  }")
  code.append("}")
  return V, batchfor_wrap(code)


def gen_blf_tensor_prods(input_indsset: List[int], output_indsset: List[int], prods: List[Tuple[int, int, int]], tens_sizes):
  code = []
  V = SharedVars()
  code.append("{ // compute left derivative tensor products")
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    for i in range(tens_sizes[inds_l]):
      code.append(f"  float accum{label}{ilabel(i)} = 0.0;")
    code.append(f"  for (int idx_chan_in{label} = threadIdx.x; idx_chan_in{label} < dim_{inds_r}; idx_chan_in{label} += blockDim.x) {{")
    code.append(f"    for (int idx_chan_out{label} = 0; idx_chan_out{label} < dim_{inds_o}; idx_chan_out{label} += 1) {{")
    code.extend(tab(tab(tab(
      gen_tensprod(tens_sizes, prod,
        LocalRef(f"float l{label}"),
        IndexRef(f"x_{inds_r}", ["idx_batch", f"idx_chan_in{label}"], ["batch", f"dim_{inds_r}"]),
        IndexRef(f"dy_{inds_o}", ["idx_batch", f"idx_chan_out{label}"], ["batch", f"dim_{inds_o}"]),
        result=0)
    ))))
    P_index = at([f"idx_chan_out{label}", f"threadIdx.y", f"idx_chan_in{label}"], [f"dim_{inds_o}", "blockDim.y", f"dim_{inds_r}"])
    code.append(f"      float P_oi{label} = P{label}{P_index};")
    for i in range(tens_sizes[inds_l]):
      code.append(f"      accum{label}{ilabel(i)} += P_oi{label}*l{label}{ilabel(i)};")
    code.append("    }")
    code.append("  }")
    code.extend(tab(warp_sum([f"accum{label}{ilabel(i)}" for i in range(tens_sizes[inds_l])])))
    code.append("  if (threadIdx.x == 0) {")
    for i in range(tens_sizes[inds_l]):
      dleft_index = at(["idx_batch", "threadIdx.y", i], ["batch", "dim_l", tens_sizes[inds_l]])
      code.append(f"    dleft{label}{dleft_index} = accum{label}{ilabel(i)};")
    code.append("  }")
  code.append("}")
  return V, batchfor_wrap(code)

def gen_wtb_tensor_prods(input_indsset: List[int], output_indsset: List[int], prods: List[Tuple[int, int, int]], tens_sizes):
  code = []
  V = SharedVars()
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    code.append(f"for (int idx_chan_in{label} = blockIdx.x; idx_chan_in{label} < dim_{inds_r}; idx_chan_in{label} += gridDim.x) {{")
    code.append(f"  for (int idx_chan_out{label} = blockIdx.y; idx_chan_out{label} < dim_{inds_o}; idx_chan_out{label} += gridDim.y) {{")
    code.append(f"    float dP_oi = 0.0;")
    code.append(f"    for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {{")
    tensprod = gen_tensprod(tens_sizes, prod,
        IndexRef(f"left{label}", ["idx_batch", "threadIdx.y"], ["batch", "dim_l"]),
        IndexRef(f"x_{inds_r}", ["idx_batch", f"idx_chan_in{label}"], ["batch", f"dim_{inds_r}"]),
        IndexRef(f"dy_{inds_o}", ["idx_batch", f"idx_chan_out{label}"], ["batch", f"dim_{inds_o}"]),
        result=None)
    code.append(f"      dP_oi += {tensprod};")
    code.append("    }")
    code.extend(tab(tab(warp_sum(["dP_oi"]))))
    code.append("    if (threadIdx.x == 0) {")
    dP_index = at([f"idx_chan_out{label}", f"threadIdx.y", f"idx_chan_in{label}"], [f"dim_{inds_o}", "blockDim.y", f"dim_{inds_r}"])
    code.append(f"      dP{label}{dP_index} = dP_oi;")
    code.append("    }")
    code.append("  }")
    code.append("}")
  return V, "\n".join(code)


def tensor_prods(name: str, input_indsset: List[int], output_indsset: List[int], prods: List[Tuple[int, int, int]], dim_l: int):
  """ Generate code for a fused kernel that does many kinds of tensor products at once.
      We also generate a corresponding backwards kernel at the same time.
      name: name to be assigned to the generated function
      input_indsset, out_indsset: list of inds's that appear as inputs and outputs respectively
      prods: list of tensor products to be computed in the form (inds_left, inds_right, inds_output)
      dim_l: dimension that the left side of each product is transformed to before tensprods are taken """
  for inds_o in output_indsset:
    assert inds_o in input_indsset, "Unsupported to have output tensorkind that is not amongst input tensorkinds."
  tens_sizes = [3**p for p in range(max(input_indsset) + 1)]
  int_params = ["batch", "dim_l", *[f"dim_{i}" for i in input_indsset]]
  # shapes setup
  shapes_fwd = {}
  shapes_bwd = {}
  shapes_blf = {}
  shapes_wtb = {}
  # define x, y params and outputs
  tens_params_fwd = [f"x_{i}" for i in input_indsset]
  tens_params_bwd = [f"dy_{i}" for i in output_indsset]
  tens_params_blf = [f"x_{i}" for i in input_indsset] + [f"dy_{i}" for i in output_indsset]
  tens_params_wtb = [f"x_{i}" for i in input_indsset] + [f"dy_{i}" for i in output_indsset]
  tens_outputs_fwd = [f"y_{i}" for i in output_indsset]
  tens_outputs_bwd = [f"dx_{i}" for i in input_indsset]
  tens_outputs_blf = []
  tens_outputs_wtb = []
  for i in input_indsset:
    shapes_fwd[f"x_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes_bwd[f"dx_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes_blf[f"x_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes_wtb[f"x_{i}"] = ("batch", f"dim_{i}") + (3,)*i
  for i in output_indsset:
    shapes_fwd[f"y_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes_bwd[f"dy_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes_blf[f"dy_{i}"] = ("batch", f"dim_{i}") + (3,)*i
    shapes_wtb[f"dy_{i}"] = ("batch", f"dim_{i}") + (3,)*i
  chunksz_exprs = {"dim_l": "dim_l"}
  chunksz_exprs.update({
    f"dim_{i}": f"dim_{i}"
    for i in input_indsset
  })
  chunksz_exprs.update({
    f"p_{i}": f"dim_l*dim_{i}"
    for i in input_indsset
  })
  gridsz_expr = "batch"
  blocksz_expr = "WARPSZ, dim_l"
  gridsz_expr_wtb = "WARPSZ, WARPSZ" # dim_j will probably be divisible by WARPSZ
  for prod in prods:
    inds_l, inds_r, inds_o = prod
    label = prodlabel(prod)
    assert inds_l in input_indsset and inds_r in input_indsset, "left or right indices outside of input set"
    assert inds_o in output_indsset, "out indices outside of output set"
    tens_params_fwd.append(f"P{label}")
    tens_params_bwd.append(f"P{label}")
    tens_params_blf.append(f"P{label}")
    tens_outputs_wtb.append(f"dP{label}")
    shapes_fwd[f"P{label}"] = (f"dim_{inds_o}", "dim_l", f"dim_{inds_r}")
    shapes_bwd[f"P{label}"] = (f"dim_{inds_o}", "dim_l", f"dim_{inds_r}")
    shapes_blf[f"P{label}"] = (f"dim_{inds_o}", "dim_l", f"dim_{inds_r}")
    shapes_wtb[f"dP{label}"] = (f"dim_{inds_o}", "dim_l", f"dim_{inds_r}")
    tens_params_fwd.append(f"left{label}")
    tens_params_bwd.append(f"left{label}")
    tens_outputs_blf.append(f"dleft{label}")
    tens_params_wtb.append(f"left{label}")
    shapes_fwd[f"left{label}"] = ("batch", "dim_l") + (3,)*inds_l
    shapes_bwd[f"left{label}"] = ("batch", "dim_l") + (3,)*inds_l
    shapes_blf[f"dleft{label}"] = ("batch", "dim_l") + (3,)*inds_l
    shapes_wtb[f"left{label}"] = ("batch", "dim_l") + (3,)*inds_l
  # Now for the hard part: generating the main computational part of the kernel!
  V_fwd, cu_fwd = gen_fwd_tensor_prods(input_indsset, output_indsset, prods, tens_sizes)
  V_bwd, cu_bwd = gen_bwd_tensor_prods(input_indsset, output_indsset, prods, tens_sizes)
  V_blf, cu_blf = gen_blf_tensor_prods(input_indsset, output_indsset, prods, tens_sizes)
  V_wtb, cu_wtb = gen_wtb_tensor_prods(input_indsset, output_indsset, prods, tens_sizes)
  return (
    Function(name, int_params, tens_params_fwd, tens_outputs_fwd, shapes_fwd, gridsz_expr, blocksz_expr, V_fwd, chunksz_exprs, cu_fwd),
    Function(name + "_backward", int_params, tens_params_bwd, tens_outputs_bwd, shapes_bwd, gridsz_expr, blocksz_expr, V_bwd, chunksz_exprs, cu_bwd),
    Function(name + "_backleft", int_params, tens_params_blf, tens_outputs_blf, shapes_blf, gridsz_expr, blocksz_expr, V_blf, chunksz_exprs, cu_blf),
    Function(name + "_wtsback", int_params, tens_params_wtb, tens_outputs_wtb, shapes_wtb, gridsz_expr_wtb, blocksz_expr, V_wtb, chunksz_exprs, cu_wtb)
  )


def bindings(functions, extra_bindings=None):
  if extra_bindings is None: extra_bindings = []
  return "\n".join([
    "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {",
    *tab([
      function.get_binding()
      for function in functions
    ]),
    *tab(extra_bindings),
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
    Fs = tensor_prods("fused_tensor_prods_example", [0, 1], [0, 1], [(0, 0, 0), (0, 1, 1), (1, 1, 0)], 8)
    for F in Fs:
      print()
      print(F.define_kern())
      print()
      print(F.define())
      print()
      print(F.define_cpp())
      print()
    print(bindings(Fs))
