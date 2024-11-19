import torch
from torch_scatter import segment_coo

import neighbour_grid_cuda as neighbour_grid

from utils import must_be


class Graph:
  """ Graph class. Supports batch of graphs which each have the same number of nodes,
      but may have different connectivity. Graphs can be symmetric or directed.
      Since total number of edges may differ between graphs, edges are stored in a
      flat tensor that indexes into a dimension of shape batch*N. """
  def __init__(self, src, dst, edge_indices, batch, nodes):
    """ src, dst: (edges)
        elems are node indices in range(0, batch*nodes). forall i src[i] must be in increasing order """
    self.src = src.to(torch.int64) # torch-scatter wants int64s
    self.dst = dst.to(torch.int64) # torch-scatter wants int64s
    edge_indices = edge_indices
    self.batch = batch
    self.nodes = nodes
  @staticmethod
  def radius_graph(r0, box, pos, celllist_max=32, neighbours_max=64):
    """ Return an instance of graph where nodes are connected if they are within r0 of each other.
        Boundary conditions are periodic, defined by box: tuple(float, float, float).
        pos: (batch, nodes, 3) is an array of node positions. """
    batch, nodes, must_be[3] = pos.shape
    src, dst, edge_indices = neighbour_grid.get_edges(celllist_max, neighbours_max, r0, *box, pos)
    ans = Graph(src, dst, edge_indices, batch, nodes)
    setattr(ans, "box", box) # since graph was created from a periodic box, we should record this
    return ans


def edges_read(graph:Graph, x_node:torch.Tensor):
  """ x_node: (batch, nodes, ...)
      ans: tuple(x_src, x_dst)
      x_src, x_dst: (edges, ...) """
  must_be[graph.batch], must_be[graph.nodes], *rest = x_node.shape
  x_node = x_node.reshape(graph.batch*graph.nodes, *rest)
  x_src = x_node[graph.src]
  x_dst = x_node[graph.dst]
  return x_src, x_dst

def edges_read_dst(graph:Graph, x_node:torch.Tensor):
  """ x_node: (batch, nodes, ...)
      x_dst: (edges, ...) """
  must_be[graph.batch], must_be[graph.nodes], *rest = x_node.shape
  x_node = x_node.reshape(graph.batch*graph.nodes, *rest)
  return x_node[graph.dst]

def edges_reduce_src(graph:Graph, x_edge:torch.Tensor):
  """ sum data on graph from edges to src nodes
      x_edge: (edges, ...) """
  ans = segment_coo(x_edge, graph.src, dim_size=graph.batch*graph.nodes)
  must_be[graph.batch*graph.nodes], *rest = ans.shape
  return ans.reshape(graph.batch, graph.nodes, *rest)


def boxwrap(box:torch.Tensor, delta_pos:torch.Tensor):
  """ take a difference in positions in a periodic box and wrap it to the shortest possible displacement
      box: (3)
      delta_pos: (..., 3) """
  return (delta_pos + 0.5*box)%box - 0.5*box


if __name__ == "__main__":
  print("Testing graph layers.")
  # create a set of positions of nodes
  positions = torch.tensor([
    [  # batch 0
      [0., 0., -1.], # node 0
      [1., 0., 0.],  # node 1
      [3., 0., 0.],  # node 2
      [2., 1., 0.],  # node 3
      [8., 0., 0.],  # node 4
    ]
  ], device="cuda")

  # create graph with radius 2. - this should connect nodes (0,1) and (1,2,3)
  graph = Graph.radius_graph(2., (100., 100., 100.), positions)

  # check src elements are in increasing order
  assert torch.all(graph.src[1:] >= graph.src[:-1])

  # test edges_read and edges_reduce_src by computing node degrees
  ones = torch.ones_like(graph.dst, dtype=torch.float)
  degrees = edges_reduce_src(graph, ones)
  expected_degrees = torch.tensor([[1., 3., 2., 2., 0.]], device="cuda")
  assert torch.allclose(degrees, expected_degrees)

  # test edges_read and edges_reduce with feature tensor
  x = torch.arange(5, device="cuda").to(torch.float).repeat(4).reshape(1, 5, 4)
  src, dst = edges_read(graph, x)
  prod = src*dst
  sumprod = edges_reduce_src(graph, prod)
  expected_sumprod = torch.tensor([ # calculated on paper!
    [0., 0., 2., 6.],
    [20., 0., 6., 8.],
    [18., 12., 0., 2.],
    [14., 12., 4., 0.],
    [0., 0., 0., 0.]
  ], device="cuda")[None]
  assert torch.allclose(sumprod, expected_sumprod)

  # test boxwrap
  pos_src, pos_dst = edges_read(graph, positions)
  delta_pos = boxwrap(torch.tensor((100., 100., 100.), device="cuda"), pos_dst - pos_src)
  assert torch.all(delta_pos[0]**2 < 2.001**2)

  print("All tests passed!")
