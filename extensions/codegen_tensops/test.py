import torch
from codegen_tensops_cuda import fused_tensor_prods_example_cuda

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman") # TODO: should probably make this project into a package at some point...


def avg_relative_diff(a, b, show_as_plot=False):
  assert a.shape == b.shape
  y = (torch.abs(a - b)/torch.sqrt(1e-12 + (a**2 + b**2))).mean(0).detach().cpu().numpy()
  if show_as_plot:
    import matplotlib.pyplot as plt
    plt.imshow(y)
    plt.show()
  return y.mean()


def refimpl(x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222):
  left_000 = torch.einsum("bi, oi -> bo", x_0, W_000)
  left_011 = torch.einsum("bi, oi -> bo", x_0, W_011)
  left_101 = torch.einsum("bij, oi -> boj", x_1, W_101)
  left_110 = torch.einsum("bij, oi -> boj", x_1, W_110)
  left_220 = torch.einsum("bijk, oi -> bojk", x_2, W_220)
  left_222 = torch.einsum("bijk, oi -> bojk", x_2, W_222)
  left_211 = torch.einsum("bijk, oi -> bojk", x_2, W_211)
  p_000 = torch.einsum("bl, br, olr -> bo", left_000, x_0, P_000)
  p_011 = torch.einsum("bl, brj, olr -> boj", left_011, x_1, P_011)
  p_101 = torch.einsum("blj, br, olr -> boj", left_101, x_0, P_101)
  p_110 = torch.einsum("blj, brj, olr -> bo", left_110, x_1, P_110)
  p_220 = torch.einsum("bljk, brjk, olr -> bo", left_220, x_2, P_220)
  p_222 = torch.einsum("bljk, brmk, olr -> bojm", left_222, x_2, P_222)
  p_211 = torch.einsum("bljk, brk, olr -> boj", left_211, x_1, P_211)
  return p_000 + p_110 + p_220, p_011 + p_101 + p_211, p_222


batch = 10
dim_0 = 64
dim_1 = 40
dim_2 = 36
dim_l = 8
# TODO: put requires grad on all inputs (`, requires_grad=True`)
W_000 = torch.randn(dim_l, dim_0, device="cuda")
W_011 = torch.randn(dim_l, dim_0, device="cuda")
W_101 = torch.randn(dim_l, dim_1, device="cuda")
W_110 = torch.randn(dim_l, dim_1, device="cuda")
W_220 = torch.randn(dim_l, dim_2, device="cuda")
W_222 = torch.randn(dim_l, dim_2, device="cuda")
W_211 = torch.randn(dim_l, dim_2, device="cuda")
P_000 = torch.randn(dim_0, dim_l, dim_0, device="cuda")
P_011 = torch.randn(dim_1, dim_l, dim_1, device="cuda")
P_101 = torch.randn(dim_1, dim_l, dim_0, device="cuda")
P_110 = torch.randn(dim_0, dim_l, dim_1, device="cuda")
P_220 = torch.randn(dim_0, dim_l, dim_2, device="cuda")
P_222 = torch.randn(dim_2, dim_l, dim_2, device="cuda")
P_211 = torch.randn(dim_1, dim_l, dim_1, device="cuda")
x_0 = torch.randn(batch, dim_0, device="cuda")
x_1 = torch.randn(batch, dim_1, 3, device="cuda")
x_2 = torch.randn(batch, dim_2, 3, 3, device="cuda")


# burn-in:
for i in range(10):
  y_0_refimpl, y_1_refimpl, y_2_refimpl = refimpl(x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222)
  y_0_ckernel, y_1_ckernel, y_2_ckernel = fused_tensor_prods_example_cuda(x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222)


# Profile CUDA kernel implementation
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True
) as prof_cuda:
    dummy = torch.randn(1000, 1000, device="cuda")
    y_0_ckernel, y_1_ckernel, y_2_ckernel = fused_tensor_prods_example_cuda(x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222)
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
    y_0_refimpl, y_1_refimpl, y_2_refimpl = refimpl(x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222)
prof_ref.export_chrome_trace("traces/refimpl_trace.json")


print(avg_relative_diff(y_0_refimpl, y_0_ckernel))
print(avg_relative_diff(y_1_refimpl, y_1_ckernel))
print(avg_relative_diff(y_2_refimpl, y_2_ckernel))
