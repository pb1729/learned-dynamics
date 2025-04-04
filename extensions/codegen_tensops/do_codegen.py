from codegen import *


functions = [tensor_prods("fused_tensor_prods_example", 2, [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 2, 0), (2, 2, 2), (2, 1, 1)], 8)]


with open("codegen_tensops.cpp", "w") as f:
  f.write(PRELUDE_CPP)
  f.write("\n\n")
  for function in functions:
    f.write(function.predeclare())
    f.write("\n\n")
  for function in functions:
    f.write(function.define_cpp())
    f.write("\n\n")
  f.write(bindings(functions))
  f.write("\n\n")

with open("codegen_tensops_kern.cu", "w") as f:
  f.write(PRELUDE_CU)
  f.write("\n\n")
  for function in functions:
    f.write(function.define_kern())
    f.write("\n\n")
    f.write(function.define())
    f.write("\n\n")
