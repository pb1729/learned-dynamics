import torch
import triton
import triton.language as tl
from typing import Any

from ..utils import must_be, avg_relative_diff


# Q: THIS FILE IS UNUSED, WHY IS IT IN THE REPO?
# A: ANY COMPLEX SYSTEM THAT WORKS MUST HAVE ARISEN FROM A SIMPLE SYSTEM THAT WORKS.
#    *THIS* IS THE SIMPLE SYSTEM THAT WORKS.


def refimpl_tensor_prod_attention(Q, K, V):
    """ reference implementation in torch of a reductionless attention operation.
        the `Z` dimension encompasses both the "batch" and "head" dimensions!
        Q, K, V: (Z, N, dim) """
    Z, N, dim = Q.shape
    *must_be[Z, N, dim], = K.shape
    *must_be[Z, N, dim], = V.shape
    QK = torch.einsum("zmd, znd -> zmn", Q, K)
    return torch.einsum("zmn, znd -> zmd", QK, V)


@triton.jit
def _fwd_kernel(
    Q, K, V,
    stride_z, stride_n,
    Out,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """ forward kernel for attention without a reduce.
        grid shape is [Z, cdiv(N, BLOCK_N)] """
    # setup block axes:
    ax_m = tl.arange(0, BLOCK_N)
    ax_n = tl.arange(0, BLOCK_N)
    ax_d = tl.arange(0, BLOCK_D)
    # identify location in grid
    z = tl.program_id(0)
    start_m = BLOCK_N*tl.program_id(1)
    idx_m = start_m + ax_m # (BLOCK_N)
    # create qkv pointers (BLOCK_N, BLOCK_D)
    q_ptrs = Q + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    k_ptrs = K + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    v_ptrs = V + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    # initialize accumulator and q block
    acc = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    q = tl.load(q_ptrs, mask=(idx_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
    # main accumulation loop
    for start_n in tl.range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        qk = tl.dot(q, tl.trans(k)) # (BLOCK_N, BLOCK_N)
        qk = tl.where(start_n + ax_n[None, :] < N, qk, 0.) # remove the out-of-range guys
        v = tl.load(v_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        acc += tl.dot(qk, v)
    out_ptrs = Out + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    tl.store(out_ptrs, acc, mask=(idx_m[:, None] < N))

@triton.jit
def _bwd_kernel_kv(
    Q, K, V, dOut,
    stride_z, stride_n,
    dK, dV,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # setup block axes:
    ax_m = tl.arange(0, BLOCK_N)
    ax_n = tl.arange(0, BLOCK_N)
    ax_d = tl.arange(0, BLOCK_D)
    # identify location in grid
    z = tl.program_id(0)
    start_n = BLOCK_N*tl.program_id(1)
    idx_n = start_n + ax_n # (BLOCK_N)
    # create qkv pointers (BLOCK_N, BLOCK_D)
    q_ptrs = Q + z*stride_z + ax_m[:, None]*stride_n + ax_d[None, :] # does not account for start_m
    k_ptrs = K + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    v_ptrs = V + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    dout_ptrs = dOut + z*stride_z + ax_m[:, None]*stride_n + ax_d[None, :] # does not account for start_m
    # load k, v which are fixed by grid location
    k = tl.load(k_ptrs, mask=(idx_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
    v = tl.load(v_ptrs, mask=(idx_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
    # main accumulation loop for dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    for start_m in tl.range(0, N, BLOCK_N):
        start_m = tl.multiple_of(start_m, BLOCK_N)
        # compute dv
        q = tl.load(q_ptrs + start_m*stride_n, mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
        dout = tl.load(dout_ptrs + start_m*stride_n, mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
        kq = tl.dot(k, tl.trans(q)) # (BLOCK_N, BLOCK_N) n,m
        kq = tl.where(start_m + ax_m[None, :] < N, kq, 0.) # remove the out-of-range guys
        dv += tl.dot(kq, dout)
        tl.debug_barrier() # must put a debug barrier and reload since triton compiler is weird
        # compute dk
        q = tl.load(q_ptrs + start_m*stride_n, mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
        dout = tl.load(dout_ptrs + start_m*stride_n, mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
        v_dout = tl.dot(v, tl.trans(dout)) # (BLOCK_N, BLOCK_N) n,m
        v_dout = tl.where(start_m + ax_m[None, :] < N, v_dout, 0.)
        dk += tl.dot(v_dout, q)
    dv_ptrs = dV + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    tl.store(dv_ptrs, dv, mask=(idx_n[:, None] < N))
    dk_ptrs = dK + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    tl.store(dk_ptrs, dk, mask=(idx_n[:, None] < N))


@triton.jit
def _bwd_kernel_q(
    Q, K, V, dOut,
    stride_z, stride_n,
    dQ,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # setup block axes:
    ax_m = tl.arange(0, BLOCK_N)
    ax_n = tl.arange(0, BLOCK_N)
    ax_d = tl.arange(0, BLOCK_D)
    # identify location in grid
    z = tl.program_id(0)
    start_m = BLOCK_N*tl.program_id(1)
    idx_m = start_m + ax_m # (BLOCK_N)
    # create qkv pointers (BLOCK_N, BLOCK_D)
    q_ptrs = Q + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    k_ptrs = K + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    v_ptrs = V + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    dout_ptrs = dOut + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    # accumulator for grad q
    dq = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=(idx_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
    dout = tl.load(dout_ptrs, mask=(idx_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
    # main accumulation loop
    for start_n in tl.range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        v = tl.load(v_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        v_dout = tl.dot(dout, tl.trans(v))
        v_dout = tl.where(start_n + ax_n[None, :] < N, v_dout, 0.) # remove the out-of-range guys
        dq += tl.dot(v_dout, k)
    dq_ptrs = dQ + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    tl.store(dq_ptrs, dq, mask=(idx_m[:, None] < N))

class _tensor_prod_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor) -> torch.Tensor:
        """ triton kernel implementation in torch of a reductionless attention operation.
            the `Z` dimension encompasses both the "batch" and "head" dimensions!
            Q, K, V: (Z, N, dim) """
        # setup shapes
        Z, N, dim = Q.shape
        *must_be[Z, N, dim], = K.shape
        *must_be[Z, N, dim], = V.shape
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        BLOCK = 32
        assert dim in {16, 32, 64, 128}
        grid = (Z, triton.cdiv(N, BLOCK))
        out = torch.empty_like(Q)
        _fwd_kernel[grid](
            Q, K, V,
            Q.stride(0), Q.stride(1),
            out,
            N,
            BLOCK_N=BLOCK,
            BLOCK_D=dim,
        )
        ctx.save_for_backward(Q, K, V)
        ctx.BLOCK = BLOCK
        ctx.dim = dim
        ctx.grid = grid
        ctx.N = N
        return out
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        BLOCK = ctx.BLOCK
        dim = ctx.dim
        grid = ctx.grid
        N = ctx.N
        dOut, = grad_outputs
        Q, K, V = ctx.saved_tensors
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        _bwd_kernel_kv[grid](
            Q, K, V, dOut,
            Q.stride(0), Q.stride(1),
            dK, dV,
            N,
            BLOCK_N=BLOCK,
            BLOCK_D=dim,
        )
        _bwd_kernel_q[grid](
            Q, K, V, dOut,
            Q.stride(0), Q.stride(1),
            dQ,
            N,
            BLOCK_N=BLOCK,
            BLOCK_D=dim,
        )
        return dQ, dK, dV
tensor_prod_attention = _tensor_prod_attention.apply


# TESTING
if __name__ == "__main__":
    if True:
        Z = 3
        N = 61
        q = torch.randn(Z, N, 16, device="cuda")
        k, v = torch.randn_like(q), torch.randn_like(q)
        dout = torch.randn_like(q)
    else:
        q = torch.tensor([[[10., *[0.]*15], *[[0.]*16]*16, [-10., *[0.]*15]]], device="cuda")
        k = torch.clone(q)
        v = torch.tensor([[[3., *[0.]*15], *[[0.]*16]*16, [0., 3., *[0.]*14]]], device="cuda")
        #dout = torch.tensor([[[2., -2., *[0.]*14]]*18], device="cuda")
        #dout = torch.tensor([[[1.0, *[0.]*15], *[[0.]*16]*16, [0., 1.0, *[0.]*14]]], device="cuda")
        dout = torch.randn_like(q)

    # require grads
    for tens in [q, k, v]:
        tens.requires_grad = True


    refimpl_out = refimpl_tensor_prod_attention(q, k, v)
    refimpl_out.backward(dout)
    refimpl_grads = (q.grad, k.grad, v.grad)

    # reset grads for next pass
    q.grad, k.grad, v.grad = None, None, None

    triton_out = tensor_prod_attention(q, k, v)
    triton_out.backward(dout)
    triton_grads = (q.grad, k.grad, v.grad)

    print("value", avg_relative_diff(refimpl_out, triton_out))
    print("grad_q", avg_relative_diff(refimpl_grads[0], triton_grads[0]))
    print("grad_k", avg_relative_diff(refimpl_grads[1], triton_grads[1]))
    print("grad_v", avg_relative_diff(refimpl_grads[2], triton_grads[2]))
