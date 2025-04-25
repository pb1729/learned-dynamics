from codegen import *


function_groups = [
  tensor_prods("fused_tensor_prods_example", [0, 1, 2], [0, 1, 2],
    [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 2, 0), (2, 2, 2), (2, 1, 1), (1, 1, 1), (2, 1, 2)], 8),
  # Ant tensprods, dim_l = 16
  tensor_prods("ant16_o0", [0, 1, 2], [0],
    [(0, 0, 0), (1, 1, 0), (2, 2, 0)], 16),
  tensor_prods("ant16_o1", [0, 1, 2], [1],
    [(0, 1, 1), (1, 0, 1), (1, 2, 1), (2, 1, 1)], 16),
  tensor_prods("ant16_o2", [0, 1, 2], [2],
    [(0, 2, 2), (2, 0, 2), (1, 1, 2), (2, 2, 2)], 16),
  tensor_prods("ant16_oc", [1, 2], [1, 2], # chiral prods
    [(1, 1, 1), (2, 1, 2)], 16),
]
functions = []
for fns in function_groups:
  for fn in fns:
    functions.append(fn)


with open("codegen_tensops.cpp", "w") as f:
  f.write(PRELUDE_CPP)
  f.write("\n\n")
  f.write("void set_kern_attributes();")
  f.write("\n\n")
  for function in functions:
    f.write(function.predeclare())
    f.write("\n\n")
  for function in functions:
    f.write(function.define_cpp())
    f.write("\n\n")
  f.write(bindings(functions, extra_bindings=['m.def("set_kern_attributes", &set_kern_attributes, "call this to initialize the module!");']))
  f.write("\n\n")

with open("codegen_tensops_kern.cu", "w") as f:
  f.write(PRELUDE_CU)
  f.write("\n\n")
  for function in functions:
    f.write(function.define_kern())
    f.write("\n\n")
    f.write(function.define())
    f.write("\n\n")
  f.write("void set_kern_attributes() {\n")
  for function in functions:
    f.write(f"  cudaFuncSetAttribute({function.fnname}_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);\n")
  f.write("}")
