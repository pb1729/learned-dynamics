import time
from itertools import permutations

import torch

from tensor_ops_cuda import tensor_linear
from codegen_tensops_cuda import fused_tensor_prods_example_cuda, fused_tensor_prods_example_backward_cuda, fused_tensor_prods_example_backleft_cuda, fused_tensor_prods_example_wtsback_cuda
import sys
sys.path.append("../..")


# define the Levi-Civita symbol
ε_ijk = torch.zeros(3, 3, 3, device="cuda")
for i, j, k in permutations(range(3)):
  ε_ijk[i, j, k] = ((i-j)/abs(i-j))*((j-k)/abs(j-k))*((k-i)/abs(k-i))


def avg_relative_diff(a, b, show_as_plot=False):
  assert a.shape == b.shape
  y = (torch.abs(a - b)/torch.sqrt(1e-12 + (a**2 + b**2))).mean(0).detach().cpu().numpy()
  if show_as_plot:
    import matplotlib.pyplot as plt
    plt.imshow(y)
    plt.show()
  return y.mean()


@torch.compile
def refimpl(x_0, x_1, x_2, W_000, P_000, W_011, P_011, W_101, P_101, W_110, P_110, W_220, P_220, W_222, P_222, W_211, P_211, W_111, P_111, W_212, P_212):
  p_000 = torch.einsum("bi, li, br, olr -> bo", x_0, W_000, x_0, P_000)
  p_011 = torch.einsum("bi, li, brj, olr -> boj", x_0, W_011, x_1, P_011)
  p_101 = torch.einsum("bij, li, br, olr -> boj", x_1, W_101, x_0, P_101)
  p_110 = torch.einsum("bij, li, brj, olr -> bo", x_1, W_110, x_1, P_110)
  p_220 = torch.einsum("bijk, li, brjk, olr -> bo", x_2, W_220, x_2, P_220)
  p_222 = torch.einsum("bijk, li, brmk, olr -> bojm", x_2, W_222, x_2, P_222)
  p_211 = torch.einsum("bijk, li, brk, olr -> boj", x_2, W_211, x_1, P_211)
  p_111 = torch.einsum("bij, li, brk, olr, jkm -> bom", x_1, W_111, x_1, P_111, ε_ijk)
  p_212 = torch.einsum("bijk, li, brm, olr, kmn -> bojn", x_2, W_212, x_1, P_212, ε_ijk)
  return p_000 + p_110 + p_220, p_011 + p_101 + p_211 + p_111, p_222 + p_212

class FusedTensorProdsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_0, x_1, x_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212):
        ctx.save_for_backward(
          x_0, x_1, x_2,
          P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212)
        return tuple(
          fused_tensor_prods_example_cuda(
            x_0, x_1, x_2,
            P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212)
        )
    @staticmethod
    def backward(ctx, dy_0, dy_1, dy_2):
        x_0, x_1, x_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212 = ctx.saved_tensors
        dx_0, dx_1, dx_2 = fused_tensor_prods_example_backward_cuda(dy_0, dy_1, dy_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212)
        dleft_000, dleft_011, dleft_101, dleft_110, dleft_220, dleft_222, dleft_211, dleft_111, dleft_212 = fused_tensor_prods_example_backleft_cuda(x_0, x_1, x_2, dy_0, dy_1, dy_2, P_000, P_011, P_101, P_110, P_220, P_222, P_211, P_111, P_212)
        dP_000, dP_011, dP_101, dP_110, dP_220, dP_222, dP_211, dP_111, dP_212 = fused_tensor_prods_example_wtsback_cuda(x_0, x_1, x_2, dy_0, dy_1, dy_2, left_000, left_011, left_101, left_110, left_220, left_222, left_211, left_111, left_212)
        return (dx_0, dx_1, dx_2, dP_000, dleft_000, dP_011, dleft_011, dP_101, dleft_101, dP_110, dleft_110, dP_220, dleft_220, dP_222, dleft_222, dP_211, dleft_211, dP_111, dleft_111, dP_212, dleft_212)

def ckernel(x_0, x_1, x_2, W_000, P_000, W_011, P_011, W_101, P_101, W_110, P_110, W_220, P_220, W_222, P_222, W_211, P_211, W_111, P_111, W_212, P_212):
  left_000 = tensor_linear(0, W_000, x_0)
  left_011 = tensor_linear(0, W_011, x_0)
  left_101 = tensor_linear(1, W_101, x_1)
  left_110 = tensor_linear(1, W_110, x_1)
  left_111 = tensor_linear(1, W_111, x_1)
  left_220 = tensor_linear(2, W_220, x_2)
  left_222 = tensor_linear(2, W_222, x_2)
  left_211 = tensor_linear(2, W_211, x_2)
  left_212 = tensor_linear(2, W_212, x_2)
  return FusedTensorProdsFunction.apply(x_0, x_1, x_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212)


batch = 10
dim_0 = 32
dim_1 = 24
dim_2 = 16
dim_l = 8
W_000 = torch.randn(dim_l, dim_0, device="cuda", requires_grad=True)
W_011 = torch.randn(dim_l, dim_0, device="cuda", requires_grad=True)
W_101 = torch.randn(dim_l, dim_1, device="cuda", requires_grad=True)
W_110 = torch.randn(dim_l, dim_1, device="cuda", requires_grad=True)
W_220 = torch.randn(dim_l, dim_2, device="cuda", requires_grad=True)
W_222 = torch.randn(dim_l, dim_2, device="cuda", requires_grad=True)
W_211 = torch.randn(dim_l, dim_2, device="cuda", requires_grad=True)
W_111 = torch.randn(dim_l, dim_1, device="cuda", requires_grad=True)
W_212 = torch.randn(dim_l, dim_2, device="cuda", requires_grad=True)
P_000 = torch.randn(dim_0, dim_l, dim_0, device="cuda", requires_grad=True)
P_011 = torch.randn(dim_1, dim_l, dim_1, device="cuda", requires_grad=True)
P_101 = torch.randn(dim_1, dim_l, dim_0, device="cuda", requires_grad=True)
P_110 = torch.randn(dim_0, dim_l, dim_1, device="cuda", requires_grad=True)
P_220 = torch.randn(dim_0, dim_l, dim_2, device="cuda", requires_grad=True)
P_222 = torch.randn(dim_2, dim_l, dim_2, device="cuda", requires_grad=True)
P_211 = torch.randn(dim_1, dim_l, dim_1, device="cuda", requires_grad=True)
P_111 = torch.randn(dim_1, dim_l, dim_1, device="cuda", requires_grad=True)
P_212 = torch.randn(dim_2, dim_l, dim_1, device="cuda", requires_grad=True)
x_0 = torch.randn(batch, dim_0, device="cuda", requires_grad=True)
x_1 = torch.randn(batch, dim_1, 3, device="cuda", requires_grad=True)
x_2 = torch.randn(batch, dim_2, 3, 3, device="cuda", requires_grad=True)
params = [x_0, x_1, x_2, W_000, P_000, W_011, P_011, W_101, P_101, W_110, P_110, W_220, P_220, W_222, P_222, W_211, P_211, W_111, P_111, W_212, P_212]

def zero_grads():
  for param in params:
    param.grad = None


def ckernel_activity():
  zero_grads()
  y_0_ckernel, y_1_ckernel, y_2_ckernel = ckernel(x_0, x_1, x_2, W_000, P_000, W_011, P_011, W_101, P_101, W_110, P_110, W_220, P_220, W_222, P_222, W_211, P_211, W_111, P_111, W_212, P_212)
  loss = y_0_ckernel.mean() + y_1_ckernel.mean() + y_2_ckernel.mean()
  loss.backward()
  return y_0_ckernel, y_1_ckernel, y_2_ckernel

def refimpl_activity():
  zero_grads()
  y_0_refimpl, y_1_refimpl, y_2_refimpl = refimpl(x_0, x_1, x_2, W_000, P_000, W_011, P_011, W_101, P_101, W_110, P_110, W_220, P_220, W_222, P_222, W_211, P_211, W_111, P_111, W_212, P_212)
  loss = y_0_refimpl.mean() + y_1_refimpl.mean() + y_2_refimpl.mean()
  loss.backward()
  return y_0_refimpl, y_1_refimpl, y_2_refimpl

# burn-in:
print("burning in calls...")
for i in range(5):
  ckernel_activity()
  refimpl_activity()
print("done.")


# Profile CUDA kernel implementation
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True
) as prof_cuda:
    dummy = torch.randn(1000, 1000, device="cuda")
    start_time = time.time()
    y_0_ckernel, y_1_ckernel, y_2_ckernel = ckernel_activity()
    grads_ckernel = [param.grad for param in params]
    print("ckernel:", time.time() - start_time)
prof_cuda.export_chrome_trace("traces/ckernel_trace.json")


# Profile reference implementation
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True
) as prof_ref:
    dummy = torch.randn(1000, 1000, device="cuda")
    start_time = time.time()
    y_0_refimpl, y_1_refimpl, y_2_refimpl = refimpl_activity()
    grads_refimpl = [param.grad for param in params]
    print("refimpl:", time.time() - start_time)
prof_ref.export_chrome_trace("traces/refimpl_trace.json")


print("y_0", avg_relative_diff(y_0_refimpl, y_0_ckernel))
print("y_1", avg_relative_diff(y_1_refimpl, y_1_ckernel))
print("y_2", avg_relative_diff(y_2_refimpl, y_2_ckernel))
for i, (param_name, param) in enumerate(zip(
    ["x_0", "x_1", "x_2", "W_000", "P_000", "W_011", "P_011", "W_101", "P_101", "W_110", "P_110", "W_220", "P_220", "W_222", "P_222", "W_211", "P_211", "W_111", "P_111", "W_212", "P_212"], params)):
  if grads_ckernel[i] is not None:
    print("d" + param_name, avg_relative_diff(grads_refimpl[i], grads_ckernel[i]))
print()
