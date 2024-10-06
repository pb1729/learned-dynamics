import torch
import torch.nn as nn
from torch_scatter import scatter

from utils import must_be


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
