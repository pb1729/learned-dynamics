import torch
import torch.nn as nn
from torch_scatter import scatter

from utils import must_be


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
    self.W = nn.Parameter(torch.randn(dim_out, dim_in))
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


class Graph:
  def __init__(self, src, dst, n_nodes):
    device = src.device
    edges, = src.shape
    self.src = src
    self.dst = dst
    self.n_nodes = n_nodes
    self.deg = scatter(torch.ones(edges, device=device), src, dim=0, dim_size=n_nodes)
    self.norm_coeff = (1. + self.deg)**(-0.5)


class VecNodesConv(nn.Module):
  """ Pass linearly transformed messages along the edges of the graph. """
  def __init__(self, dim_in, dim_out):
    super().__init__()
    # layers:
    self.linear_node = VecLinear(dim_in, dim_out)
    self.linear_edge = VecLinear(dim_in, dim_out)
  def forward(self, x, graph):
    """ x: (batch, nodes, dim_in, 3)
        return: (batch, nodes, dim_out, 3) """
    y_node = self.linear_node(x)
    y_edge = self.linear_edge(x) # compute transformed values before doing graph convolution
    y_edge = scatter(y_edge[:, graph.src], graph.dst, dim=1, dim_size=graph.n_nodes) # pass information along edges
    ans = y_node + graph.norm_coeff[:, None, None]*y_edge
    return ans*INV_SQRT_2

class VecEdgesRead(nn.Module):
  """ edges read from nodes """
  def __init__(self, dim_in, dim_out):
    super().__init__()
    # layers:
    self.linear_src = VecLinear(dim_in, dim_out)
    self.linear_dst = VecLinear(dim_in, dim_out)
  def forward(self, x, graph):
    """ x: (batch, nodes, dim_in, 3)
        return: (batch, edges, dim_out, 3) """
    ans = (self.linear_src(x[:, graph.src]) + self.linear_dst(x[:, graph.dst]))
    return ans*INV_SQRT_2

class VecEdgesWrite(nn.Module):
  """ edges write to nodes """
  def __init__(self, dim_in, dim_out):
    super().__init__()
    # layers:
    self.linear_src = VecLinear(dim_in, dim_out)
    self.linear_dst = VecLinear(dim_in, dim_out)
  def forward(self, x, graph):
    """ x: (batch, edges, dim_in, 3)
        return: (batch, nodes, dim_out, 3) """
    ans = (scatter(self.linear_src(x), graph.src, dim=1, dim_size=graph.n_nodes)
         + scatter(self.linear_dst(x), graph.dst, dim=1, dim_size=graph.n_nodes))
    return ans*INV_SQRT_2*graph.norm_coeff[:, None, None]

class ScalNodesConv(nn.Module):
  """ Pass linearly transformed messages along the edges of the graph. """
  def __init__(self, dim_in, dim_out, bias=False):
    super().__init__()
    # layers:
    self.linear_node = nn.Linear(dim_in, dim_out, bias=bias)
    self.linear_edge = nn.Linear(dim_in, dim_out, bias=bias)
  def forward(self, x, graph):
    """ x: (batch, nodes, dim_in)
        return: (batch, nodes, dim_out) """
    y_node = self.linear_node(x)
    y_edge = self.linear_edge(x) # compute transformed values before doing graph convolution
    y_edge = scatter(y_edge[:, graph.src], graph.dst, dim=1, dim_size=graph.n_nodes) # pass information along edges
    ans = y_node + graph.norm_coeff[:, None]*y_edge
    return ans*INV_SQRT_2

class ScalEdgesRead(nn.Module):
  """ edges read from nodes """
  def __init__(self, dim_in, dim_out, bias=False):
    super().__init__()
    # layers:
    self.linear_src = nn.Linear(dim_in, dim_out, bias=bias)
    self.linear_dst = nn.Linear(dim_in, dim_out, bias=bias)
  def forward(self, x, graph):
    """ x: (batch, nodes, dim_in)
        return: (batch, edges, dim_out) """
    ans = (self.linear_src(x[:, graph.src]) + self.linear_dst(x[:, graph.dst]))
    return ans*INV_SQRT_2

class ScalEdgesWrite(nn.Module):
  """ edges write to nodes """
  def __init__(self, dim_in, dim_out, bias=False):
    super().__init__()
    # layers:
    self.linear_src = nn.Linear(dim_in, dim_out, bias=bias)
    self.linear_dst = nn.Linear(dim_in, dim_out, bias=bias)
  def forward(self, x, graph):
    """ x: (batch, edges, dim_in)
        return: (batch, nodes, dim_out) """
    ans = (scatter(self.linear_src(x), graph.src, dim=1, dim_size=graph.n_nodes)
         + scatter(self.linear_dst(x), graph.dst, dim=1, dim_size=graph.n_nodes))
    return ans*INV_SQRT_2*graph.norm_coeff[:, None]

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


class EdgeRelativeEmbed(nn.Module):
  """ input embedding for edges, where 2 positions are passed as input """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_a = nn.Linear(6, adim)
    self.lin_v = VecLinear(6, vdim)
  def forward(self, pos_0, pos_1, graph):
    """ pos0: (batch, node, 3)
        pos1: (batch, node, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, edge, adim)
        v_out: (batch, edge, vdim, 3) """
    vecs = torch.stack([ # 6 = (4 choose 2) relative vectors
        pos_0[:, graph.dst] - pos_0[:, graph.src],
        pos_1[:, graph.dst] - pos_1[:, graph.src],
        pos_1[:, graph.src] - pos_0[:, graph.src],
        pos_1[:, graph.dst] - pos_0[:, graph.dst],
        pos_1[:, graph.src] - pos_0[:, graph.dst],
        pos_1[:, graph.dst] - pos_0[:, graph.src],
      ], dim=2) # (batch, edges, 6, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, edges, 6)
    a_out = self.lin_a(norms)
    v_out = self.lin_v(vecs)/3 # fudge factor of 1/3
    return a_out, v_out


class NodeRelativeEmbed(nn.Module):
  """ input embedding for nodes, where 2 positions are passed as input """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_a = nn.Linear(1, adim)
    self.lin_v = VecLinear(1, vdim)
  def forward(self, pos_0, pos_1, graph):
    """ pos0: (batch, node, 3)
        pos1: (batch, node, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, edge, adim)
        v_out: (batch, edge, vdim, 3) """
    vecs = torch.stack([ # 1 = (2 choose 2) relative vectors
        pos_1 - pos_0,
      ], dim=2) # (batch, edges, 1, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, edges, 1)
    a_out = self.lin_a(norms)
    v_out = self.lin_v(vecs)/3 # fudge factor of 1/3
    return a_out, v_out


class EdgeRelativeEmbedMLP(nn.Module):
  """ input embedding for edges, where 2 positions are passed as input """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_v = VecLinear(6, vdim)
    self.scalar_layers = nn.Sequential(
      nn.Linear(6, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
  def forward(self, pos_0, pos_1, graph):
    """ pos0: (batch, node, 3)
        pos1: (batch, node, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, edge, adim)
        v_out: (batch, edge, vdim, 3) """
    vecs = torch.stack([ # 6 = (4 choose 2) relative vectors
        pos_0[:, graph.dst] - pos_0[:, graph.src],
        pos_1[:, graph.dst] - pos_1[:, graph.src],
        pos_1[:, graph.src] - pos_0[:, graph.src],
        pos_1[:, graph.dst] - pos_0[:, graph.dst],
        pos_1[:, graph.src] - pos_0[:, graph.dst],
        pos_1[:, graph.dst] - pos_0[:, graph.src],
      ], dim=2) # (batch, edges, 6, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, edges, 6)
    a_out = self.scalar_layers(norms)
    v_out = self.lin_v(vecs)/3 # fudge factor of 1/3
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
  def forward(self, pos_0, pos_1, graph):
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








