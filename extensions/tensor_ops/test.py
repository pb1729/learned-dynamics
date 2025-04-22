import torch
from tensor_ops_cuda import *

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman/managan") # TODO: should probably make this project into a package at some point...
from utils import must_be, avg_relative_diff


def refimpl_tensor_linear(inds, W, x):
    """ reference implementation in torch of a reductionless attention operation.
        the `Z` dimension encompasses both the "batch" and "head" dimensions!
        Q, K, V: (Z, N, dim) """
    tensor_indices = "".join([chr(ord("i") + n) for n in range(inds)]) # eg "ijk" if inds==3
    einsum_str = f"OI, ...I{tensor_indices} -> ...O{tensor_indices}"
    return torch.einsum(einsum_str, W, x)


batch = 10
dim_in = 117
dim_out = 210
W = torch.randn(dim_out, dim_in, device="cuda", requires_grad=True)


for inds in range(3):
  print("\nTesting for inds = %d" % inds)
  W.grad = None # clear gradient
  x = torch.randn(batch, dim_in, *[3]*inds, device="cuda", requires_grad=True)
  y_refimpl = refimpl_tensor_linear(inds, W, x)
  dy = torch.randn(batch, dim_out, *[3]*inds, device="cuda")
  y_refimpl.backward(dy)
  dx_refimpl = x.grad
  dW_refimpl = W.grad
  x.grad = None
  W.grad = None
  y_ckernel = tensor_linear(inds, W, x)
  y_ckernel.backward(dy)
  dx_ckernel = x.grad
  dW_ckernel = W.grad
  print("value", avg_relative_diff(y_refimpl, y_ckernel))
  print("dx", avg_relative_diff(dx_refimpl, dx_ckernel))
  print("dW", avg_relative_diff(dW_refimpl, dW_ckernel))
