import torch
import torch.nn as nn

from .utils import must_be


# constants:
INV_SQRT_2 = 0.5**0.5


def weights_init(m):
  """ custom weights initialization """
  cls = m.__class__
  if hasattr(cls, "self_init"):
    m.self_init()
    return
  classname = cls.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)
    return
  if hasattr(m, "bias") and m.bias is not None:
    nn.init.constant_(m.bias.data, 0)


class Residual(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(dim, dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(dim),
        nn.Linear(dim, dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(dim),
      )
  def forward(self, x):
    return x + self.layers(x)


class ResidualConv1d(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv1d(dim, dim, 5, padding="same"),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(dim),
        nn.Conv1d(dim, dim, 5, padding="same"),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(dim),
      )
  def forward(self, x):
    return x + self.layers(x)


class ToAtomCoords(nn.Module):
  def __init__(self, space_dim):
    super().__init__()
    self.space_dim = space_dim
  def forward(self, x):
    """ x: (batch, n_atoms*space_dim) """
    batch, state_dim = x.shape
    assert state_dim % self.space_dim == 0
    n_atoms = state_dim // self.space_dim
    y = x.reshape(batch, n_atoms, self.space_dim)
    return y.transpose(1, 2)

class FromAtomCoords(nn.Module):
  def __init__(self, space_dim):
    super().__init__()
    self.space_dim = space_dim
  def forward(self, x):
    """ x: (batch, space_dim, n_atoms) """
    batch, space_dim, n_atoms = x.shape
    assert space_dim == self.space_dim
    return x.transpose(2, 1).reshape(batch, n_atoms*space_dim)


# SO3 symmetric layers for 3D graph networks:

class VecLinear(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.W = nn.Parameter(torch.empty(dim_out, dim_in))
  def self_init(self):
    _, fan_in = self.W.shape
    nn.init.normal_(self.W, std=fan_in**-0.5)
  def forward(self, v):
    return torch.einsum("oi, ...ik -> ...ok", self.W, v)

class VecRootS(nn.Module):
  """ an activation function for vectors that is S-shaped along any given direction """
  def __init__(self):
    super().__init__()
  def forward(self, v):
    v_sq = (v**2).sum(-1, keepdim=True)
    scale = (1. + v_sq)**(-0.25)
    return v*scale

class _VectorSigmoid(torch.autograd.Function):
  """ vector activation function with a sigmoid shape in any given direction """
  @staticmethod
  def forward(ctx, v):
    """ x: (..., 3) """
    ctx.save_for_backward(v)
    vv = (v**2).sum(-1, keepdim=True)
    return v/torch.sqrt(1. + vv)
  @staticmethod
  def backward(ctx, grad_output):
    v, = ctx.saved_tensors
    vv = (v**2).sum(-1, keepdim=True)
    quad = 1. + vv
    return (quad*grad_output - (v*grad_output).sum(-1, keepdim=True)*v)*(quad**-1.5)
vector_sigmoid = _VectorSigmoid.apply

class VectorSigmoid(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return vector_sigmoid(x)

def radial_encode(r, n, rmax):
  """ r: (...)
      ans: (..., n)"""
  n = torch.arange(0, n, device=r.device)
  return torch.cos(torch.pi*n*r[..., None])


class VecResidual(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.layers = nn.Sequential(
      VecLinear(dim, dim),
      VecRootS(),
      VecLinear(dim, dim),
      VecRootS(),
      VecLinear(dim, dim))
  def forward(self, x):
    return x + self.layers(x)


class ScalVecProducts(nn.Module):
  """ computes dot, cross, and scalar products, respects SO3 symmetry
      product has a reduced rank, and is divided by a normalization
      factor of sqrt(1 + x**2 + y**2) """
  def __init__(self, adim, vdim, rank):
    super().__init__()
    self.left_lin_a  = nn.Linear(adim, rank)
    self.right_lin_a = nn.Linear(adim, rank)
    self.left_lin_v  = VecLinear(vdim, rank)
    self.right_lin_v = VecLinear(vdim, rank)
    self.aa_lin_a = nn.Linear(rank, adim)
    self.vv_lin_a = nn.Linear(rank, adim)
    self.av_lin_v = VecLinear(rank, vdim)
    self.vv_lin_v = VecLinear(rank, vdim)
  def forward(self, a, v):
    """ a: (..., adim)
        v: (..., vdim, 3)
        return: tuple(a_out, v_out)
        a_out: (..., adim)
        v_out: (..., vdim, 3) """
    a_left, a_right = self.left_lin_a(a), self.right_lin_a(a)
    v_left, v_right = self.left_lin_v(v), self.right_lin_v(v)
    a_left_sq, a_right_sq = a_left**2, a_right**2
    v_left_sq, v_right_sq = (v_left**2).sum(-1), (v_right**2).sum(-1)
    aa_a = (a_left*a_right)                    / torch.sqrt(1. + a_left_sq + a_right_sq)
    vv_a = (v_left*v_right).sum(-1)            / torch.sqrt(1. + v_left_sq + v_right_sq)
    av_v = (a_left[..., None]*v_right)         / torch.sqrt(1. + a_left_sq + v_right_sq)[..., None]
    vv_v = torch.linalg.cross(v_left, v_right) / torch.sqrt(1. + v_left_sq + v_right_sq)[..., None]
    a_out = self.aa_lin_a(aa_a) + self.vv_lin_a(vv_a)
    v_out = self.av_lin_v(av_v) + self.vv_lin_v(vv_v)
    return a_out, v_out


class NodeRelativeEmbedMLP(nn.Module):
  """ input embedding for nodes, where 2 positions are passed as input """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_v = VecLinear(1, vdim)
    self.scalar_layers = nn.Sequential(
      nn.Linear(1, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
  def forward(self, pos_0, pos_1):
    """ pos0: (batch, node, 3)
        pos1: (batch, node, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, edge, adim)
        v_out: (batch, edge, vdim, 3) """
    vecs = torch.stack([ # 1 = (2 choose 2) relative vectors
        pos_1 - pos_0,
      ], dim=2) # (batch, edges, 1, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, edges, 1)
    a_out = self.scalar_layers(norms)
    v_out = self.lin_v(vecs)/3 # fudge factor of 1/3
    return a_out, v_out


class ScalGroupNorm(nn.Module):
  """ Scalar group norm, should be a fairly standard group norm as found eg here:
      https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
      We average over the nodes (or edges) of the graph as well as each group. """
  def __init__(self, chan, groups, epsilon=1e-5):
    super().__init__()
    assert chan % groups == 0
    self.chan = chan
    self.groups = groups
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(chan))
    self.beta = nn.Parameter(torch.zeros(chan))
  def forward(self, x):
    """ note: where we write "nodes" here, we could also write "edges"
        x: (batch, nodes, chan)
        return: (batch, nodes, chan) """
    batch, nodes, must_be[self.chan] = x.shape
    x = x.reshape(batch, nodes, self.groups, -1)
    mean = x.mean([1, 3], keepdim=True)
    x_shift = x - mean
    var = ((x_shift)**2).mean([1, 3], keepdim=True)
    ans = x_shift/torch.sqrt(self.epsilon + var)
    return self.beta + self.gamma*ans.reshape(batch, nodes, self.chan)

class VecGroupNorm(nn.Module):
  def __init__(self, chan, groups, epsilon=1e-5):
    super().__init__()
    assert chan % groups == 0
    self.chan = chan
    self.groups = groups
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(chan, 1))
  def forward(self, x):
    """ note: where we write "nodes" here, we could also write "edges"
        x: (batch, nodes, chan, 3)
        return: (batch, nodes, chan, 3) """
    batch, nodes, must_be[self.chan], must_be[3] = x.shape
    x = x.reshape(batch, nodes, self.groups, -1, 3)
    moment2 = (x**2).sum(4, keepdim=True).mean([1, 3], keepdim=True) # (batch, 1, groups, 1, 1)
    ans = x/torch.sqrt(self.epsilon + moment2)
    return self.gamma*ans.reshape(batch, nodes, self.chan, 3)



class ScalConv1d(nn.Module):
  def __init__(self, chan, kernsz, edges_to_nodes=False, dilation=1):
    super().__init__()
    if edges_to_nodes:
      assert kernsz % 2 == 0
      assert dilation == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, dilation=dilation, padding=(kernsz//2))
    else:
      assert kernsz % 2 == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, dilation=dilation, padding="same")
  def forward(self, x):
    """ x: (batch, length, chan) """
    x = x.permute(0, 2, 1) # (batch, chan, length)
    y = self.conv(x)
    y = y.permute(0, 2, 1) # (batch, newlength, chan)
    return y

class VecConv1d(nn.Module):
  def __init__(self, chan, kernsz, edges_to_nodes=False, dilation=1):
    super().__init__()
    if edges_to_nodes:
      assert kernsz % 2 == 0
      assert dilation == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, dilation=dilation, padding=(kernsz//2), bias=False)
    else:
      assert kernsz % 2 == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, dilation=dilation, padding="same", bias=False)
  def forward(self, x):
    """ x: (batch, length, chan, 3) """
    batch, length, chan, must_be[3] = x.shape
    x = x.permute(0, 3, 2, 1).reshape(3*batch, chan, length) # (batch*3, chan, length)
    y = self.conv(x)
    must_be[batch*3], must_be[chan], newlength = y.shape # length might have changed!
    y = y.reshape(batch, 3, chan, newlength).permute(0, 3, 2, 1) # (batch, newlength, chan, 3)
    return y


class EdgeRelativeEmbedMLPPath(nn.Module):
  """ input embedding for edges, where 2 positions are passed as input.
      assumes graph structure is path """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_v = VecLinear(4, vdim)
    self.scalar_layers = nn.Sequential(
      nn.Linear(4, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.conv_v = VecConv1d(vdim, 4, True)
    self.conv_a = ScalConv1d(adim, 4, True)
  def forward(self, pos_0, pos_1):
    """ pos0: (batch, nodes, 3)
        pos1: (batch, nodes, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, nodes, adim)
        v_out: (batch, nodes, vdim, 3) """
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    vecs = torch.stack([ # 4 relative vectors
        pos_0[:,  1:] - pos_0[:, :-1],
        pos_1[:,  1:] - pos_1[:, :-1],
        pos_1[:, :-1] - pos_0[:,  1:],
        pos_1[:,  1:] - pos_0[:, :-1],
      ], dim=2) # (batch, nodes - 1, 4, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, nodes - 1, 4)
    y_a = self.scalar_layers(norms)
    y_v = self.lin_v(vecs)
    a_out = self.conv_a(y_a)
    v_out = self.conv_v(y_v)
    return a_out, v_out


class IndexedLinear(nn.Module): # TODO: test it out, and add bias
  """ Linear layer with many weight matrices. Which weight matrix is used is indexed by an int tensor. """
  def __init__(self, ch_in, ch_out, n):
    super().__init__()
    self.n = n
    self.W = nn.Parameter(torch.empty(n, ch_out, ch_in))
  def self_init(self):
    must_be[self.n], fan_in, fan_out = self.W.shape
    nn.init.normal_(self.W, std=fan_in**-0.5)
  def forward(self, x, sel):
    """ apply indexed linear to x, with weight matrices selected by sel
        x: (batch, nodes, ch_in)
        sel: (nodes) -- type is int. we use the same selection across the batch
        ans: (batch nodes, ch_out) """
    return torch.einsum("bnj, nij -> bni", x, self.W[sel])


def periodic_rel(pos_0, pos_1, L): # TODO: test it out
  """ get translation-invariant periodicity-respecting relative position vector
      pos_0: (..., 3)
      pos_1: (..., 3)
      L: (3) -- box periodicity
      ans: (..., 3) """
  d_pos = pos_1 - pos_0
  return (L*0.15915494309189535)*torch.sin(d_pos*6.283185307179586/L)

def periodic_sqrel(pos_0, pos_1, L): # TODO: test it out
  """ get translation-invariant periodicity-respecting squared relative position vector
      pos_0: (..., 3)
      pos_1: (..., 3)
      L: (3) -- box periodicity
      ans: (...) """
  d_pos = pos_1 - pos_0
  return (((L*0.3183098861837907)*torch.sin(d_pos*3.141592653589793/L))**2).sum(-1)
