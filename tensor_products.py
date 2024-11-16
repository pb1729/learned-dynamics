from itertools import permutations

import torch
import torch.nn as nn

from utils import must_be
from layers_common import VecLinear


class TensLinear(nn.Module):
  def __init__(self, inds, dim_in, dim_out):
    super().__init__()
    self.W = nn.Parameter(torch.empty(dim_out, dim_in))
    tensor_indices = "".join([chr(ord("i") + n) for n in range(inds)]) # eg "ijk" if inds==3
    self.einsum_str = f"OI, ...I{tensor_indices} -> ...O{tensor_indices}"
    self.self_init()
  def self_init(self):
    _, fan_in = self.W.shape
    nn.init.normal_(self.W, std=fan_in**-0.5)
  def forward(self, x):
    return torch.einsum(self.einsum_str, self.W, x)

# define the Levi-Civita symbol {
ε_ijk = torch.zeros(3, 3, 3, device="cuda")
for i, j, k in permutations(range(3)):
  ε_ijk[i, j, k] = ((i-j)/abs(i-j))*((j-k)/abs(j-k))*((k-i)/abs(k-i))
# }
def ε_d2v(x_d):
  return torch.einsum("...ij, ijk -> ...k", x_d, ε_ijk)
def ε_v2d(x_v):
  return torch.einsum("...i, ijk -> ...kj", x_v, ε_ijk)
class ChiralMix(nn.Module):
  """ Linear module that acts on inputs x_v with 1 index and x_d with 2 indices.
      Introduce chirality by contracting with Levi-Civita symbol. """
  def __init__(self, dim_v, dim_d):
    super().__init__()
    self.lin_vd = VecLinear(dim_v, dim_d)
    self.lin_dv = VecLinear(dim_d, dim_v)
  def forward(self, x_v, x_d):
    return self.lin_dv(ε_d2v(x_d)), ε_v2d(self.lin_vd(x_v))

class TensTrans(nn.Module):
  """ Linear module that acts on a tensor with 2 of its spatial dims transposed. """
  def __init__(self, inds, idx_0, idx_1, dim_in, dim_out):
    super().__init__()
    assert 0 <= idx_0 < inds, "index 0 out of range"
    assert 0 <= idx_1 < inds, "index 1 out of range"
    assert idx_0 != idx_1, "can't swap dim with itself"
    self.lin = TensLinear(inds, dim_in, dim_out)
    self.idx_0 = idx_0 - inds
    self.idx_1 = idx_1 - inds
  def forward(self, x):
    return self.lin(torch.transpose(x, self.idx_0, self.idx_1))

class TensTrace(nn.Module):
  """ Linear module that applies trace. """
  def __init__(self, inds, dim_in, dim_out):
    super().__init__()
    assert inds >= 2
    self.lin = TensLinear(inds - 2, dim_in, dim_out)
  def forward(self, x):
    return self.lin(torch.einsum("...jj -> ...", x))

identity = torch.eye(3, device="cuda")
class TensDelta(nn.Module):
  """ Linear module that applies delta_ij. """
  def __init__(self, inds, dim_in, dim_out):
    super().__init__()
    self.lin = TensLinear(inds, dim_in, dim_out)
  def forward(self, x):
    return self.lin(x)[..., None, None]*identity


class TensorProds(nn.Module):
  def __init__(self, inds_l:int, inds_r:int, inds_o:int, dim_l:int, dim_r:int, dim_o:int, rank:int):
    super().__init__()
    def assert_div_2(a):
      assert a%2 == 0, "parity mismatch"
      return a//2
    contracts = assert_div_2(inds_l + inds_r - inds_o) # number of tensor contractions
    assert 0 <= contracts <= min(inds_l, inds_r), f"{contracts} contractions is out of range for {inds_l}, {inds_r} index tensors"
    self.lin_l = TensLinear(inds_l, dim_l, rank)
    self.lin_r = TensLinear(inds_r, dim_r, rank)
    self.lin_o = TensLinear(inds_o, rank, dim_o)
    tensor_indices = "".join([chr(ord("i") + n) for n in range(inds_l + inds_r - contracts)])
    to_contract = tensor_indices[:contracts]
    keep_l      = tensor_indices[contracts:inds_l]
    keep_r      = tensor_indices[inds_l:]
    self.einsum_str = f"...{keep_l + to_contract}, ...{keep_r + to_contract} -> ...{keep_l + keep_r}"
  def forward(self, x_l, x_r):
    x_l = self.lin_l(x_l)
    x_r = self.lin_r(x_r)
    x_o = torch.einsum(self.einsum_str, x_l, x_r)
    return self.lin_o(x_o)


class TensorRandGen:
  """ A class the generated random torch tensors with a SO(3)-symmetric
      probability distribution. Number of spatial indices of the tensor
      is arbitrary. Can also specify a non-spatial shape, of course. By
      setting a seed can get reproducible results. These repeated results
      can be transformed using set_transform. """
  def __init__(self):
    self.transform = None
  def set_transform(self, transform):
    """ transform: indices, (..., (3,)^indices) -> (..., (3,)^indices) """
    self.transform = transform
  def clear_transform(self):
    self.transform = None
  def randn(self, spatial_indices, shape, device="cuda", dtype=torch.float32):
    ans = torch.randn(*shape, *[3]*spatial_indices, device=device, dtype=dtype)
    if self.transform is not None:
      ans = self.transform(spatial_indices, ans)
    return ans


class TensConv1d(nn.Module):
  """ 1d convolution of arbitrary tensor """
  def __init__(self, inds, chan, kernsz, dilation=1):
    super().__init__()
    assert kernsz % 2 == 1, "kernel size must be odd"
    self.inds = inds
    self.conv = nn.Conv1d(chan, chan, kernsz, dilation=dilation, padding="same", bias=False)
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    """ x: (batch, length, chan, (3,)^inds) """
    batch, length, chan, *must_be[(3,)*self.inds] = x.shape
    x = x.reshape(batch, length, chan, 3**self.inds)
    x = x.transpose(1, 3) # (batch, 3^inds, chan, length)
    x = x.reshape((3**self.inds)*batch, chan, length) # prep shape for 1d conv
    y = self.conv(x)
    must_be[(3**self.inds)*batch], must_be[chan], newlength = y.shape # length might have changed!
    y = y.reshape(batch, 3**self.inds, chan, newlength)# (batch, 3^inds, chan, newlength)
    y = y.transpose(1, 3) # (batch, newlength, chan, 3^inds)
    y = y.reshape(batch, length, chan, *[3]*self.inds) # expand tensor back out
    return y


class _TensSigmoid(torch.autograd.Function):
  """ tensor activation function with a sigmoid shape """
  @staticmethod
  def forward(ctx, inds, x):
    """ x: (..., (3,)^inds) """
    ctx.save_for_backward(x)
    ctx.dimtup = tuple([-(1 + n) for n in range(inds)])
    xx = (x**2).sum(ctx.dimtup, keepdim=True)
    return x/torch.sqrt(1. + xx)
  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    xx = (x**2).sum(ctx.dimtup, keepdim=True)
    quad = 1. + xx
    return None, (quad*grad_output - (x*grad_output).sum(ctx.dimtup, keepdim=True)*x)*(quad**-1.5)
tens_sigmoid = _TensSigmoid.apply


class TensGroupNorm(nn.Module):
  def __init__(self, inds, chan, groups, epsilon=1e-5):
    super().__init__()
    assert chan % groups == 0
    self.inds = inds
    self.dimtup = tuple([-(1 + n) for n in range(inds)])
    self.chan = chan
    self.groups = groups
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(chan, *[1]*inds))
  def forward(self, x):
    """ note: where we write "nodes" here, we could also write "edges"
        x: (batch, nodes, chan, (3,)^inds)
        return: (batch, nodes, chan, (3,)^inds) """
    batch, nodes, must_be[self.chan], *must_be[(3,)*self.inds] = x.shape
    x = x.reshape(batch, nodes, self.groups, -1, *[3]*self.inds)
    moment2 = (x**2).sum(self.dimtup, keepdim=True).mean([1, 3], keepdim=True) # (batch, 1, groups, 1, (1,)^inds)
    ans = x/torch.sqrt(self.epsilon + moment2)
    return self.gamma*ans.reshape(batch, nodes, self.chan, *[3]*self.inds)


class AVDFullLinearMix(nn.Module):
  """ Full linear mixture for tensors up to 2 indices. """
  def __init__(self, dim_a, dim_v, dim_d):
    super().__init__()
    self.ttrans = TensTrans(2, 0, 1, dim_d, dim_d)
    self.tdelta = TensDelta(0, dim_a, dim_d)
    self.ttrace = TensTrace(2, dim_d, dim_a)
    self.chiral = ChiralMix(dim_v, dim_d)
  def forward(self, x_a, x_v, x_d):
    x_d = x_d + self.ttrans(x_d)
    x_a, x_d = x_a + self.ttrace(x_d), x_d + self.tdelta(x_a)
    Δx_v, Δx_d = self.chiral(x_v, x_d)
    x_v, x_d = x_v + Δx_v, x_d + Δx_d
    return x_a, x_v, x_d

class AVDFullTensorProds(nn.Module):
  def __init__(self, dim_a, dim_v, dim_d, rank):
    super().__init__()
    self.tp000 = TensorProds(0, 0, 0, dim_a, dim_a, dim_a, rank)
    self.tp101 = TensorProds(1, 0, 1, dim_v, dim_a, dim_v, rank)
    self.tp110 = TensorProds(1, 1, 0, dim_v, dim_v, dim_a, rank)
    self.tp112 = TensorProds(1, 1, 2, dim_v, dim_v, dim_d, rank)
    self.tp202 = TensorProds(2, 0, 2, dim_d, dim_a, dim_d, rank)
    self.tp211 = TensorProds(2, 1, 1, dim_d, dim_v, dim_v, rank)
    self.tp220 = TensorProds(2, 2, 0, dim_d, dim_d, dim_a, rank)
    self.tp222 = TensorProds(2, 2, 2, dim_d, dim_d, dim_d, rank)
  def forward(self, x_a, x_v, x_d):
    y_a = self.tp000(x_a, x_a) + self.tp110(x_v, x_v) + self.tp220(x_d, x_d)
    y_v = self.tp101(x_v, x_a) + self.tp211(x_d, x_v)
    y_d = self.tp112(x_v, x_v) + self.tp202(x_d, x_a) + self.tp222(x_d, x_d)
    return y_a, y_v, y_d




if __name__ == "__main__":
  print(ε_ijk)
