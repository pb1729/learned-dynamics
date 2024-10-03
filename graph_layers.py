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
