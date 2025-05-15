import torch

from codegen_tensops_cuda import (set_kern_attributes,
  ant16_o0_cuda, ant16_o0_backward_cuda, ant16_o0_backleft_cuda, ant16_o0_wtsback_cuda,
  ant16_o1_cuda, ant16_o1_backward_cuda, ant16_o1_backleft_cuda, ant16_o1_wtsback_cuda,
  ant16_o2_cuda, ant16_o2_backward_cuda, ant16_o2_backleft_cuda, ant16_o2_wtsback_cuda,
  ant16_oc_cuda, ant16_oc_backward_cuda, ant16_oc_backleft_cuda, ant16_oc_wtsback_cuda,
  bee_fwd_cuda, bee_bwl_cuda, bee_bwr_cuda
)

# initialize the module (so that we have enough shared memory)
#set_kern_attributes()
# above is currently commented because we only need it to pass wide tensors to Ant16. Since
# the most current models use Bee, it would just result in needless incompatibility with systems with older GPUs.


class Ant16(torch.autograd.Function):
  @staticmethod
  def forward(ctx,
      x_0, x_1, x_2,
      P_000, left_000, P_110, left_110, P_220, left_220,
      P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211, P_111, left_111,
      P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222, P_212, left_212):
    ctx.save_for_backward(x_0, x_1, x_2,
      P_000, left_000, P_110, left_110, P_220, left_220,
      P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211, P_111, left_111,
      P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222, P_212, left_212)
    y_0, = ant16_o0_cuda(x_0, x_1, x_2, P_000, left_000, P_110, left_110, P_220, left_220)
    y_1, = ant16_o1_cuda(x_0, x_1, x_2, P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211)
    y_2, = ant16_o2_cuda(x_0, x_1, x_2, P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222)
    y_1c, y_2c = ant16_oc_cuda(x_1, x_2, P_111, left_111, P_212, left_212)
    return y_0, y_1 + y_1c, y_2 + y_2c
  @staticmethod
  def backward(ctx, dy_0, dy_1, dy_2):
    (x_0, x_1, x_2,
      P_000, left_000, P_110, left_110, P_220, left_220,
      P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211, P_111, left_111,
      P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222, P_212, left_212
    ) = ctx.saved_tensors
    dx_0_o0, dx_1_o0, dx_2_o0 = ant16_o0_backward_cuda(dy_0, P_000, left_000, P_110, left_110, P_220, left_220)
    dx_0_o1, dx_1_o1, dx_2_o1 = ant16_o1_backward_cuda(dy_1, P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211)
    dx_0_o2, dx_1_o2, dx_2_o2 = ant16_o2_backward_cuda(dy_2, P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222)
    dx_1_oc, dx_2_oc = ant16_oc_backward_cuda(dy_1, dy_2, P_111, left_111, P_212, left_212)
    dleft_000, dleft_110, dleft_220 = ant16_o0_backleft_cuda(x_0, x_1, x_2, dy_0, P_000, P_110, P_220)
    dleft_011, dleft_101, dleft_121, dleft_211 = ant16_o1_backleft_cuda(x_0, x_1, x_2, dy_1, P_011, P_101, P_121, P_211)
    dleft_022, dleft_202, dleft_112, dleft_222 = ant16_o2_backleft_cuda(x_0, x_1, x_2, dy_2, P_022, P_202, P_112, P_222)
    dleft_111, dleft_212 = ant16_oc_backleft_cuda(x_1, x_2, dy_1, dy_2, P_111, P_212)
    dP_000, dP_110, dP_220 = ant16_o0_wtsback_cuda(x_0, x_1, x_2, dy_0, left_000, left_110, left_220)
    dP_011, dP_101, dP_121, dP_211 = ant16_o1_wtsback_cuda(x_0, x_1, x_2, dy_1, left_011, left_101, left_121, left_211)
    dP_022, dP_202, dP_112, dP_222 = ant16_o2_wtsback_cuda(x_0, x_1, x_2, dy_2, left_022, left_202, left_112, left_222)
    dP_111, dP_212 = ant16_oc_wtsback_cuda(x_1, x_2, dy_1, dy_2, left_111, left_212)
    return (
      dx_0_o0 + dx_0_o1 + dx_0_o2,
      dx_1_o0 + dx_1_o1 + dx_1_o2 + dx_1_oc,
      dx_2_o0 + dx_2_o1 + dx_2_o2 + dx_2_oc,
      dP_000, dleft_000, dP_110, dleft_110, dP_220, dleft_220,
      dP_011, dleft_011, dP_101, dleft_101, dP_121, dleft_121, dP_211, dleft_211, dP_111, dleft_111,
      dP_022, dleft_022, dP_202, dleft_202, dP_112, dleft_112, dP_222, dleft_222, dP_212, dleft_212
    )


class Bee(torch.autograd.Function):
  @staticmethod
  def forward(ctx, l_0, l_1, l_2, r_0, r_1, r_2):
    ctx.save_for_backward(l_0, l_1, l_2, r_0, r_1, r_2)
    y_000, y_110, y_220, y_011, y_101, y_121, y_211, y_022, y_202, y_112, y_222, y_111, y_212 = bee_fwd_cuda(l_0, l_1, l_2, r_0, r_1, r_2)
    return y_000, y_110, y_220, y_011, y_101, y_121, y_211, y_022, y_202, y_112, y_222, y_111, y_212
  @staticmethod
  def backward(ctx, dy_000, dy_110, dy_220, dy_011, dy_101, dy_121, dy_211, dy_022, dy_202, dy_112, dy_222, dy_111, dy_212):
    l_0, l_1, l_2, r_0, r_1, r_2 = ctx.saved_tensors
    dl_0, dl_1, dl_2 = bee_bwl_cuda(r_0, r_1, r_2, dy_000, dy_110, dy_220, dy_011, dy_101, dy_121, dy_211, dy_022, dy_202, dy_112, dy_222, dy_111, dy_212)
    dr_0, dr_1, dr_2 = bee_bwr_cuda(l_0, l_1, l_2, dy_000, dy_110, dy_220, dy_011, dy_101, dy_121, dy_211, dy_022, dy_202, dy_112, dy_222, dy_111, dy_212)
    return dl_0, dl_1, dl_2, dr_0, dr_1, dr_2


if __name__ == "__main__":
  from itertools import permutations
  from utils import avg_relative_diff
  # define the Levi-Civita symbol
  lc_ijk = torch.zeros(3, 3, 3, device="cuda")
  for i, j, k in permutations(range(3)):
    lc_ijk[i, j, k] = ((i-j)/abs(i-j))*((j-k)/abs(j-k))*((k-i)/abs(k-i))
  def ant16_refimpl(x_0, x_1, x_2,
      P_000, left_000, P_110, left_110, P_220, left_220,
      P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211, P_111, left_111,
      P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222, P_212, left_212):
    p_000 = torch.einsum("bl, br, olr -> bo", left_000, x_0, P_000)
    p_110 = torch.einsum("blx, brx, olr -> bo", left_110, x_1, P_110)
    p_220 = torch.einsum("blxy, brxy, olr -> bo", left_220, x_2, P_220)
    p_011 = torch.einsum("bl, brx, olr -> box", left_011, x_1, P_011)
    p_101 = torch.einsum("blx, br, olr -> box", left_101, x_0, P_101)
    p_121 = torch.einsum("blx, bryx, olr -> boy", left_121, x_2, P_121)
    p_211 = torch.einsum("blxy, bry, olr -> box", left_211, x_1, P_211)
    p_111 = torch.einsum("blx, bry, olr, xyz -> boz", left_111, x_1, P_111, lc_ijk)
    p_022 = torch.einsum("bl, brxy, olr -> boxy", left_022, x_2, P_022)
    p_202 = torch.einsum("blxy, br, olr -> boxy", left_202, x_0, P_202)
    p_112 = torch.einsum("blx, bry, olr -> boxy", left_112, x_1, P_112)
    p_222 = torch.einsum("blxy, brzy, olr -> boxz", left_222, x_2, P_222)
    p_212 = torch.einsum("blxy, brz, olr, yzw -> boxw", left_212, x_1, P_212, lc_ijk)
    return (
      p_000 + p_110 + p_220,
      p_011 + p_101 + p_121 + p_211 + p_111,
      p_022 + p_202 + p_112 + p_222 + p_212
    )
  print("TEST: ANT16")
  batch = 10
  dim_0 = 64
  dim_1 = 48
  dim_2 = 32
  dim_l = 16
  left_000 = torch.randn(batch, dim_l, device="cuda", requires_grad=True)
  left_110 = torch.randn(batch, dim_l, 3, device="cuda", requires_grad=True)
  left_220 = torch.randn(batch, dim_l, 3, 3, device="cuda", requires_grad=True)
  left_011 = torch.randn(batch, dim_l, device="cuda", requires_grad=True)
  left_101 = torch.randn(batch, dim_l, 3, device="cuda", requires_grad=True)
  left_121 = torch.randn(batch, dim_l, 3, device="cuda", requires_grad=True)
  left_211 = torch.randn(batch, dim_l, 3, 3, device="cuda", requires_grad=True)
  left_111 = torch.randn(batch, dim_l, 3, device="cuda", requires_grad=True)
  left_022 = torch.randn(batch, dim_l, device="cuda", requires_grad=True)
  left_202 = torch.randn(batch, dim_l, 3, 3, device="cuda", requires_grad=True)
  left_112 = torch.randn(batch, dim_l, 3, device="cuda", requires_grad=True)
  left_222 = torch.randn(batch, dim_l, 3, 3, device="cuda", requires_grad=True)
  left_212 = torch.randn(batch, dim_l, 3, 3, device="cuda", requires_grad=True)
  P_000 = torch.randn(dim_0, dim_l, dim_0, device="cuda", requires_grad=True)
  P_110 = torch.randn(dim_0, dim_l, dim_1, device="cuda", requires_grad=True)
  P_220 = torch.randn(dim_0, dim_l, dim_2, device="cuda", requires_grad=True)
  P_011 = torch.randn(dim_1, dim_l, dim_1, device="cuda", requires_grad=True)
  P_101 = torch.randn(dim_1, dim_l, dim_0, device="cuda", requires_grad=True)
  P_121 = torch.randn(dim_1, dim_l, dim_2, device="cuda", requires_grad=True)
  P_211 = torch.randn(dim_1, dim_l, dim_1, device="cuda", requires_grad=True)
  P_111 = torch.randn(dim_1, dim_l, dim_1, device="cuda", requires_grad=True)
  P_022 = torch.randn(dim_2, dim_l, dim_2, device="cuda", requires_grad=True)
  P_202 = torch.randn(dim_2, dim_l, dim_0, device="cuda", requires_grad=True)
  P_112 = torch.randn(dim_2, dim_l, dim_1, device="cuda", requires_grad=True)
  P_222 = torch.randn(dim_2, dim_l, dim_2, device="cuda", requires_grad=True)
  P_212 = torch.randn(dim_2, dim_l, dim_1, device="cuda", requires_grad=True)
  x_0 = torch.randn(batch, dim_0, device="cuda", requires_grad=True)
  x_1 = torch.randn(batch, dim_1, 3, device="cuda", requires_grad=True)
  x_2 = torch.randn(batch, dim_2, 3, 3, device="cuda", requires_grad=True)
  params = [
    x_0, x_1, x_2,
    left_000, P_000, left_110, P_110, left_220, P_220,
    left_011, P_011, left_101, P_101, left_121, P_121, left_211, P_211, left_111, P_111,
    left_022, P_022, left_202, P_202, left_112, P_112, left_222, P_222, left_212, P_212
  ]
  param_names = [
    "x_0", "x_1", "x_2",
    "left_000", "P_000", "left_110", "P_110", "left_220", "P_220",
    "left_011", "P_011", "left_101", "P_101", "left_121", "P_121", "left_211", "P_211", "left_111", "P_111",
    "left_022", "P_022", "left_202", "P_202", "left_112", "P_112", "left_222", "P_222", "left_212", "P_212"
  ]
  def zero_grad():
    for param in params:
      param.grad = None
  # Do the test:
  for i in range(10): # burn-in
    # call reference implementation
    zero_grad()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
      asdf = torch.randn(300, 600, 3, device="cuda") # let the profiler warm up...
      y0_ref, y1_ref, y2_ref = ant16_refimpl(
        x_0, x_1, x_2,
        P_000, left_000, P_110, left_110, P_220, left_220,
        P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211, P_111, left_111,
        P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222, P_212, left_212)
      loss = y0_ref.mean() + y1_ref.mean() + y2_ref.mean()
      loss.backward()
    prof.export_chrome_trace("../traces/refimpl.json")
    grads_ref = [param.grad for param in params]
    # call optimized implementation
    zero_grad()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
      asdf = torch.randn(300, 600, 3, device="cuda") # let the profiler warm up...
      y0, y1, y2 = Ant16.apply(
        x_0, x_1, x_2,
        P_000, left_000, P_110, left_110, P_220, left_220,
        P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211, P_111, left_111,
        P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222, P_212, left_212)
      loss = y0.mean() + y1.mean() + y2.mean()
      loss.backward()
    prof.export_chrome_trace("../traces/ckernel.json")
    grads = [param.grad for param in params]

  # verify outputs match
  print("Forward pass comparison:")
  print(f"y0 avg_relative_diff: {avg_relative_diff(y0_ref, y0)}")
  print(f"y1 avg_relative_diff: {avg_relative_diff(y1_ref, y1)}")
  print(f"y2 avg_relative_diff: {avg_relative_diff(y2_ref, y2)}")
  for param_name, grad_ref, grad in zip(param_names, grads_ref, grads):
    print(f"d{param_name} avg_relative_diff: {avg_relative_diff(grad_ref, grad)}")

  print("TEST: BEE")
  def bee_refimpl(l_0, l_1, l_2, r_0, r_1, r_2):
    y_000 = torch.einsum("bc, bc -> bc", l_0, r_0)
    y_110 = torch.einsum("bcx, bcx -> bc", l_1, r_1)
    y_220 = torch.einsum("bcxy, bcxy -> bc", l_2, r_2)
    y_011 = torch.einsum("bc, bcx -> bcx", l_0, r_1)
    y_101 = torch.einsum("bcx, bc -> bcx", l_1, r_0)
    y_121 = torch.einsum("bcx, bcyx -> bcy", l_1, r_2)
    y_211 = torch.einsum("bcxy, bcy -> bcx", l_2, r_1)
    y_111 = torch.einsum("bcx, bcy, xyz -> bcz", l_1, r_1, lc_ijk)
    y_022 = torch.einsum("bc, bcxy -> bcxy", l_0, r_2)
    y_202 = torch.einsum("bcxy, bc -> bcxy", l_2, r_0)
    y_112 = torch.einsum("bcx, bcy -> bcxy", l_1, r_1)
    y_222 = torch.einsum("bcxy, bczy -> bcxz", l_2, r_2)
    y_212 = torch.einsum("bcxy, bcz, yzw -> bcxw", l_2, r_1, lc_ijk)
    return y_000, y_110, y_220, y_011, y_101, y_121, y_211, y_022, y_202, y_112, y_222, y_111, y_212
  batch = 10
  chan = 256
  l_0 = torch.randn(batch, chan, device="cuda", requires_grad=True)
  l_1 = torch.randn(batch, chan, 3, device="cuda", requires_grad=True)
  l_2 = torch.randn(batch, chan, 3, 3, device="cuda", requires_grad=True)
  r_0 = torch.randn(batch, chan, device="cuda", requires_grad=True)
  r_1 = torch.randn(batch, chan, 3, device="cuda", requires_grad=True)
  r_2 = torch.randn(batch, chan, 3, 3, device="cuda", requires_grad=True)
  params = [l_0, l_1, l_2, r_0, r_1, r_2]
  param_names = ["l_0", "l_1", "l_2", "r_0", "r_1", "r_2"]
  def zero_grad_bee():
    for param in params:
      param.grad = None
  y_refimpl = bee_refimpl(l_0, l_1, l_2, r_0, r_1, r_2)
  grad_y = [torch.randn_like(y_p) for y_p in y_refimpl]
  loss_refimpl = sum([(y_p*grad_y_p).mean() for y_p, grad_y_p in zip(y_refimpl, grad_y)])
  loss_refimpl.backward()
  grads_refimpl = [param.grad for param in params]
  zero_grad_bee()
  y_ckernel = Bee.apply(l_0, l_1, l_2, r_0, r_1, r_2)
  loss_ckernel = sum([(y_p*grad_y_p).mean() for y_p, grad_y_p in zip(y_ckernel, grad_y)])
  loss_ckernel.backward()
  grads_ckernel = [param.grad for param in params]
  for i, output_name in enumerate("y_000, y_110, y_220, y_011, y_101, y_121, y_211, y_022, y_202, y_112, y_222, y_111, y_212".split(", ")):
    print(f"{output_name} avg_relative_diff: {avg_relative_diff(y_refimpl[i], y_ckernel[i])}")
  for param_name, grad_ref, grad in zip(param_names, grads_refimpl, grads_ckernel):
    print(f"d{param_name} avg_relative_diff: {avg_relative_diff(grad_ref, grad)}")
