import torch
import triton
import triton.language as tl
from typing import Any

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman") # TODO: should probably make this project into a package at some point...
from utils import must_be, avg_relative_diff


def refimpl_proxattn(Q, K, V, X, Y, r0sq):
    """ reference implementation in torch of a reductionless attention operation.
        the `Z` dimension encompasses both the "batch" and "head" dimensions!
        Q, K, V: (Z, N, dim)
        X, Y: (Z, N, 3)
        r0sq: (heads) """
    Z, N, dim = Q.shape
    *must_be[Z, N, dim], = K.shape
    *must_be[Z, N, dim], = V.shape
    *must_be[Z, N, 3], = X.shape
    *must_be[Z, N, 3], = Y.shape
    heads, = r0sq.shape
    assert Z % heads == 0
    r0sq = (r0sq[None].expand(Z//heads, -1)).reshape(Z)
    QK = torch.einsum("zmd, znd -> zmn", Q, K)
    R = X[:, :, None] - Y[:, None, :]
    W = r0sq[:, None, None]*QK/(r0sq[:, None, None] + (R**2).sum(-1))
    return torch.einsum("zmn, znd -> zmd", W, V)


@triton.jit
def _fwd_kernel(
    Q, K, V, X, Y, R0SQ,
    stride_z, stride_n,
    stride_pz, stride_pn,
    Out,
    N, heads,
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
    r0sq = tl.load(R0SQ + (z % heads))
    # create qkv pointers (BLOCK_N, BLOCK_D) and xy pointers (BLOCK_N)
    q_ptrs = Q + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    k_ptrs = K + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    v_ptrs = V + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    x_ptrs = X + z*stride_pz + idx_m[:, None]*stride_pn
    y_ptrs = Y + z*stride_pz + ax_n[None, :]*stride_pn # does not account for start_n
    # initialize q block and x blocks
    q = tl.load(q_ptrs,      mask=(idx_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
    x0 = tl.load(x_ptrs + 0, mask=(idx_m[:, None] < N))
    x1 = tl.load(x_ptrs + 1, mask=(idx_m[:, None] < N))
    x2 = tl.load(x_ptrs + 2, mask=(idx_m[:, None] < N))
    # main accumulation loop
    acc = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # compute qk
        k = tl.load(k_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        qk = tl.dot(q, tl.trans(k)) # (BLOCK_N, BLOCK_N) m,n
        qk = tl.where(start_n + ax_n[None, :] < N, qk, 0.) # remove the out-of-range guys
        # geometry stuff
        y0 = tl.load(y_ptrs + start_n*stride_pn + 0, mask=(start_n + ax_n[None, :] < N))
        y1 = tl.load(y_ptrs + start_n*stride_pn + 1, mask=(start_n + ax_n[None, :] < N))
        y2 = tl.load(y_ptrs + start_n*stride_pn + 2, mask=(start_n + ax_n[None, :] < N))
        r0 = x0 - y0 # (BLOCK_N, BLOCK_N) m,n
        r1 = x1 - y1 # (BLOCK_N, BLOCK_N) m,n
        r2 = x2 - y2 # (BLOCK_N, BLOCK_N) m,n
        rsq = r0*r0 + r1*r1 + r2*r2 # (BLOCK_N, BLOCK_N) m,n
        # compute weights and accumulate
        W = r0sq*qk/(r0sq + rsq)
        v = tl.load(v_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        acc += tl.dot(W, v)
    out_ptrs = Out + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    tl.store(out_ptrs, acc, mask=(idx_m[:, None] < N))


@triton.jit
def _bwd_kernel_kvy(
    Q, K, V, X, Y, R0SQ, dOut,
    stride_z, stride_n,
    stride_pz, stride_pn,
    dK, dV, dY,
    N, heads,
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
    r0sq = tl.load(R0SQ + (z % heads))
    # create qkv pointers (BLOCK_N, BLOCK_D) and xy pointers (BLOCK_N)
    q_ptrs = Q + z*stride_z + ax_m[:, None]*stride_n + ax_d[None, :] # does not account for start_m
    k_ptrs = K + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    v_ptrs = V + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    x_ptrs = X + z*stride_pz + ax_m[None, :]*stride_pn # does not account for start_m
    y_ptrs = Y + z*stride_pz + idx_n[:, None]*stride_pn
    dout_ptrs = dOut + z*stride_z + ax_m[:, None]*stride_n + ax_d[None, :] # does not account for start_m
    # load k, v, y which are fixed by grid location
    k = tl.load(k_ptrs,      mask=(idx_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
    v = tl.load(v_ptrs,      mask=(idx_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
    y0 = tl.load(y_ptrs + 0, mask=(idx_n[:, None] < N)) # (BLOCK_N, 1)
    y1 = tl.load(y_ptrs + 1, mask=(idx_n[:, None] < N)) # (BLOCK_N, 1)
    y2 = tl.load(y_ptrs + 2, mask=(idx_n[:, None] < N)) # (BLOCK_N, 1)
    # main accumulation loop for dv and dk and dx
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dy0 = tl.zeros([BLOCK_N], dtype=tl.float32)
    dy1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    dy2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_m in tl.range(0, N, BLOCK_N):
        start_m = tl.multiple_of(start_m, BLOCK_N)
        # geometry stuff
        x0 = tl.load(x_ptrs + start_m*stride_pn + 0, mask=(start_m + ax_m[None, :] < N)) # (1, BLOCK_N)
        x1 = tl.load(x_ptrs + start_m*stride_pn + 1, mask=(start_m + ax_m[None, :] < N)) # (1, BLOCK_N)
        x2 = tl.load(x_ptrs + start_m*stride_pn + 2, mask=(start_m + ax_m[None, :] < N)) # (1, BLOCK_N)
        r0 = x0 - y0 # (BLOCK_N, BLOCK_N) n,m
        r1 = x1 - y1 # (BLOCK_N, BLOCK_N) n,m
        r2 = x2 - y2 # (BLOCK_N, BLOCK_N) n,m
        rsq = r0*r0 + r1*r1 + r2*r2 # (BLOCK_N, BLOCK_N) n,m
        denom = 1./(r0sq + rsq) # (BLOCK_N, BLOCK_N) n,m
        # compute dv
        q = tl.load(q_ptrs + start_m*stride_n,       mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
        dout = tl.load(dout_ptrs + start_m*stride_n, mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
        kq = tl.dot(k, tl.trans(q)) # (BLOCK_N, BLOCK_N) n,m
        kq = tl.where(start_m + ax_m[None, :] < N, kq, 0.) # remove the out-of-range guys
        dv += tl.dot(kq*r0sq*denom, dout)
        # compute dk
        tl.debug_barrier() # must put a debug barrier since triton compiler is weird
        q = tl.load(q_ptrs + start_m*stride_n,       mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D) must reload since triton compiler is weird
        dout = tl.load(dout_ptrs + start_m*stride_n, mask=(start_m + ax_m[:, None] < N)) # (BLOCK_N, BLOCK_D) must reload since triton compiler is weird
        dW = tl.dot(v, tl.trans(dout)) # (BLOCK_N, BLOCK_N) n,m
        dW = tl.where(start_m + ax_m[None, :] < N, dW, 0.) # remove the out of range guys
        dk += tl.dot(dW*r0sq*denom, q)
        # y derivative
        tl.debug_barrier() # must put a debug barrier since triton compiler is weird
        coeff_nm = dW*kq*r0sq*denom*denom # (BLOCK_N, BLOCK_N) m,n
        coeff_nm = tl.where(start_m + ax_m[None, :] < N, coeff_nm, 0.) # remove the out-of-range guys
        dy0 += tl.sum(2*r0*coeff_nm, axis=1)
        dy1 += tl.sum(2*r1*coeff_nm, axis=1)
        dy2 += tl.sum(2*r2*coeff_nm, axis=1)
    dv_ptrs = dV + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    tl.store(dv_ptrs, dv, mask=(idx_n[:, None] < N))
    dk_ptrs = dK + z*stride_z + idx_n[:, None]*stride_n + ax_d[None, :]
    tl.store(dk_ptrs, dk, mask=(idx_n[:, None] < N))
    dy0_ptrs = dY + z*stride_pz + idx_n*stride_pn + 0
    dy1_ptrs = dY + z*stride_pz + idx_n*stride_pn + 1
    dy2_ptrs = dY + z*stride_pz + idx_n*stride_pn + 2
    tl.store(dy0_ptrs, dy0, mask=(idx_n < N))
    tl.store(dy1_ptrs, dy1, mask=(idx_n < N))
    tl.store(dy2_ptrs, dy2, mask=(idx_n < N))


@triton.jit
def _bwd_kernel_qx(
    Q, K, V, X, Y, R0SQ, dOut,
    stride_z, stride_n,
    stride_pz, stride_pn,
    dQ, dX,
    N, heads,
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
    r0sq = tl.load(R0SQ + (z % heads))
    # create qkv pointers (BLOCK_N, BLOCK_D) and xy pointers (BLOCK_N)
    q_ptrs = Q + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    k_ptrs = K + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    v_ptrs = V + z*stride_z + ax_n[:, None]*stride_n + ax_d[None, :] # does not account for start_n
    x_ptrs = X + z*stride_pz + idx_m[:, None]*stride_pn
    y_ptrs = Y + z*stride_pz + ax_n[None, :]*stride_pn # does not account for start_n
    dout_ptrs = dOut + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    # load m indexed tensors
    q = tl.load(q_ptrs,       mask=(idx_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
    x0 = tl.load(x_ptrs + 0,  mask=(idx_m[:, None] < N)) # (BLOCK_N, 1)
    x1 = tl.load(x_ptrs + 1,  mask=(idx_m[:, None] < N)) # (BLOCK_N, 1)
    x2 = tl.load(x_ptrs + 2,  mask=(idx_m[:, None] < N)) # (BLOCK_N, 1)
    dout = tl.load(dout_ptrs, mask=(idx_m[:, None] < N)) # (BLOCK_N, BLOCK_D)
    # main accumulation loop
    dq = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dx0 = tl.zeros([BLOCK_N], dtype=tl.float32)
    dx1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    dx2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_n in tl.range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        v = tl.load(v_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D)
        dW = tl.dot(dout, tl.trans(v)) # (BLOCK_N, BLOCK_N) m,n
        dW = tl.where(start_n + ax_n[None, :] < N, dW, 0.) # remove the out-of-range guys
        # geometry stuff
        y0 = tl.load(y_ptrs + start_n*stride_pn + 0, mask=(start_n + ax_n[None, :] < N)) # (1, BLOCK_N)
        y1 = tl.load(y_ptrs + start_n*stride_pn + 1, mask=(start_n + ax_n[None, :] < N)) # (1, BLOCK_N)
        y2 = tl.load(y_ptrs + start_n*stride_pn + 2, mask=(start_n + ax_n[None, :] < N)) # (1, BLOCK_N)
        r0 = x0 - y0 # (BLOCK_N, BLOCK_N) m,n
        r1 = x1 - y1 # (BLOCK_N, BLOCK_N) m,n
        r2 = x2 - y2 # (BLOCK_N, BLOCK_N) m,n
        rsq = r0*r0 + r1*r1 + r2*r2 # (BLOCK_N, BLOCK_N) m,n
        denom = 1./(r0sq + rsq) # (BLOCK_N, BLOCK_N) m,n
        # q derivative
        dq += tl.dot(dW*r0sq*denom, k)
        # x derivative
        tl.debug_barrier() # must put a debug barrier since triton compiler is weird
        k = tl.load(k_ptrs + start_n*stride_n, mask=(start_n + ax_n[:, None] < N)) # (BLOCK_N, BLOCK_D) must reload since triton compiler is weird
        qk = tl.dot(q, tl.trans(k)) # (BLOCK_N, BLOCK_N) m,n
        coeff_mn = dW*qk*r0sq*denom*denom # (BLOCK_N, BLOCK_N) m,n
        coeff_mn = tl.where(start_n + ax_n[None, :] < N, coeff_mn, 0.) # remove the out-of-range guys
        dx0 += tl.sum(-2*r0*coeff_mn, axis=1)
        dx1 += tl.sum(-2*r1*coeff_mn, axis=1)
        dx2 += tl.sum(-2*r2*coeff_mn, axis=1)
    dq_ptrs = dQ + z*stride_z + idx_m[:, None]*stride_n + ax_d[None, :]
    tl.store(dq_ptrs, dq, mask=(idx_m[:, None] < N))
    dx0_ptrs = dX + z*stride_pz + idx_m*stride_pn + 0
    dx1_ptrs = dX + z*stride_pz + idx_m*stride_pn + 1
    dx2_ptrs = dX + z*stride_pz + idx_m*stride_pn + 2
    tl.store(dx0_ptrs, dx0, mask=(idx_m < N))
    tl.store(dx1_ptrs, dx1, mask=(idx_m < N))
    tl.store(dx2_ptrs, dx2, mask=(idx_m < N))


class _proxattn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, X:torch.Tensor, Y:torch.Tensor, r0sq:torch.Tensor) -> torch.Tensor:
        """ triton kernel implementation in torch of a reductionless attention operation.
            the `Z` dimension encompasses both the "batch" and "head" dimensions!
            Q, K, V: (Z, N, dim)
            x, y: (Z, N, 3) -- probe points
            r0sq: (heads) """
        # setup shapes
        Z, N, dim = Q.shape
        *must_be[Z, N, dim], = K.shape
        *must_be[Z, N, dim], = V.shape
        *must_be[Z, N, 3], = X.shape
        *must_be[Z, N, 3], = Y.shape
        heads, = r0sq.shape
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        assert X.is_contiguous() and Y.is_contiguous()
        assert r0sq.is_contiguous()
        assert Z % heads == 0
        BLOCK = 16 if dim > 64 else 32 # make sure we don't run out of shared memory
        assert dim in {16, 32, 64, 128}
        grid = (Z, triton.cdiv(N, BLOCK))
        out = torch.empty_like(Q)
        _fwd_kernel[grid](
            Q, K, V, X, Y, r0sq,
            Q.stride(0), Q.stride(1),
            X.stride(0), X.stride(1),
            out,
            N, heads,
            BLOCK_N=BLOCK,
            BLOCK_D=dim,
        )
        ctx.save_for_backward(Q, K, V, X, Y, r0sq)
        ctx.BLOCK = BLOCK
        ctx.dim = dim
        ctx.grid = grid
        ctx.N = N
        ctx.heads = heads
        return out
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        BLOCK = ctx.BLOCK
        dim = ctx.dim
        grid = ctx.grid
        N = ctx.N
        heads = ctx.heads
        dOut, = grad_outputs
        Q, K, V, X, Y, r0sq = ctx.saved_tensors
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dX = torch.empty_like(X)
        dY = torch.empty_like(Y)
        _bwd_kernel_kvy[grid](
            Q, K, V, X, Y, r0sq, dOut,
            Q.stride(0), Q.stride(1),
            X.stride(0), X.stride(1),
            dK, dV, dY,
            N, heads,
            BLOCK_N=BLOCK,
            BLOCK_D=dim,
        )
        _bwd_kernel_qx[grid](
            Q, K, V, X, Y, r0sq, dOut,
            Q.stride(0), Q.stride(1),
            X.stride(0), X.stride(1),
            dQ, dX,
            N, heads,
            BLOCK_N=BLOCK,
            BLOCK_D=dim,
        )
        return dQ, dK, dV, dX, dY, None
proxattn = _proxattn.apply


# TESTING CODE
if __name__ == "__main__":
    from clobbercheck import ClobberChecker
    if True:#with ClobberChecker() as cc:
        print("\n"*20, "STARTING TEST\n\n")
        for i in range(2100): # do lots of steps so we can see if there will be an illegal memory access (usually around 936)
            print(1 + i)
            r0sq = torch.tensor([0.2, 0.3, 0.5, 0.8, 1.3], device="cuda")
            heads, = r0sq.shape
            if True:
                Z = 7*heads
                N = 1 + i
                dim = 128
                q = torch.randn(Z, N, dim, device="cuda")
                k, v = torch.randn_like(q), torch.randn_like(q)
                x, y = 2*torch.rand(Z, N, 3, device="cuda"), 2*torch.rand(Z, N, 3, device="cuda")
                dout = torch.randn_like(q)
            else:
                q = torch.tensor([[[10., *[0.]*15], *[[0.]*16]*16, [-10., *[0.]*15]]], device="cuda")
                k = torch.clone(q)
                v = torch.tensor([[[3., *[0.]*15], *[[0.]*16]*16, [0., 3., *[0.]*14]]], device="cuda")
                #dout = torch.tensor([[[2., -2., *[0.]*14]]*18], device="cuda")
                #dout = torch.tensor([[[1.0, *[0.]*15], *[[0.]*16]*16, [0., 1.0, *[0.]*14]]], device="cuda")
                dout = torch.randn_like(q)

            # require grads
            for tens in [q, k, v, x, y]:
                tens.requires_grad = True

            refimpl_out = refimpl_proxattn(q, k, v, x, y, r0sq)
            refimpl_out.backward(dout)
            refimpl_grads = (q.grad, k.grad, v.grad, x.grad, y.grad)

            # reset grads for next pass
            q.grad, k.grad, v.grad, x.grad, y.grad = None, None, None, None, None

            triton_out = proxattn(q, k, v, x, y, r0sq)
            triton_out.backward(dout)
            #cc.report()
            triton_grads = (q.grad, k.grad, v.grad, x.grad, y.grad)

            print("value", avg_relative_diff(refimpl_out, triton_out))
            print("grad_q", avg_relative_diff(refimpl_grads[0], triton_grads[0]))
            print("grad_k", avg_relative_diff(refimpl_grads[1], triton_grads[1]))
            print("grad_v", avg_relative_diff(refimpl_grads[2], triton_grads[2]))
            print("grad_x", avg_relative_diff(refimpl_grads[3], triton_grads[3]))
            print("grad_y", avg_relative_diff(refimpl_grads[4], triton_grads[4]))
