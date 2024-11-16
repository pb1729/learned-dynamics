import torch
from torch.nested import nested_tensor
from torch_scatter import segment_coo

import neighbour_grid_cuda as neighbour_grid

from utils import must_be


class Graph:
  """ Graph class. Supports batch of graphs which have the same number of nodes,
      but may have different connectivity. Graphs can be symmetric or directed.
      Implementation uses nested tensors. So various tensors in a batch may have
      differing numbers of edges. The number of nodes is still shared, though. """
  def __init__(self, src, dst, batch, nodes):
    """ src, dst: nested(batch, (edges))
        elems are node indices in range(0, nodes). forall i src[i] must be in increasing order """
    self.src = src.to(torch.int64) # torch-scatter wants int64s
    self.dst = dst.to(torch.int64) # torch-scatter wants int64s
    self.batch = batch
    self.nodes = nodes
  @staticmethod
  def radius_graph(r0, box, pos, celllist_max=32, neighbours_max=40):
    """ Return an instance of graph where nodes are connected if they are within r0 of each other.
        Boundary conditions are periodic, defined by box: tuple(float, float, float).
        pos: (batch, nodes, 3) is an array of node positions. """
    batch, nodes, must_be[3] = pos.shape
    src, dst = neighbour_grid.get_edges(celllist_max, neighbours_max, r0, *box, pos)
    ans = Graph(src, dst, batch, nodes)
    setattr(ans, "box", box) # since graph was created from a periodic box, we should record this
    return ans


def edges_read(graph:Graph, x_node:torch.Tensor):
  """ x_node: (batch, nodes, ...)
      ans: tuple(x_src, x_dst)
      x_src, x_dst: nested(batch, (edges, ...)) """
  # nested tensors currently don't support torch.gather, but this function would be equivalent to this:
  # return torch.gather(x_node, 1, graph.src), torch.gather(x_node, 1, graph.dst)
  x_src = nested_tensor([
    x_node[i, graph.src[i]]
    for i in range(graph.batch)])
  x_dst = nested_tensor([
    x_node[i, graph.dst[i]]
    for i in range(graph.batch)])
  return x_src, x_dst

def edges_read_dst(graph:Graph, x_node:torch.Tensor):
  """ x_node: (batch, nodes, ...)
      x_dst: nested(batch, (edges, ...)) """
  # nested tensors currently don't support torch.gather, but this function would be equivalent to this:
  # torch.gather(x_node, 1, graph.dst)
  return nested_tensor([
    x_node[i, graph.dst[i]]
    for i in range(graph.batch)])

def edges_reduce_src(graph:Graph, x_edge:torch.Tensor):
  """ sum data on graph from edges to src nodes
      x_edge: nested(batch, (edges, ...)) """
  return torch.stack([
    segment_coo(x_edge[i], graph.src[i], dim_size=graph.nodes)
    for i in range(graph.batch)])

def boxwrap(box:torch.Tensor, delta_pos:torch.Tensor):
  """ take a difference in positions in a periodic box and wrap it to the shortest possible displacement
      box: (3)
      delta_pos: (..., 3) """
  if delta_pos.is_nested:
    return torch.nested.nested_tensor([
      (delta_pos[i] + 0.5*box)%box - 0.5*box
      for i in range(delta_pos.size(0))])
  else:
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
  for i in range(graph.batch):
    assert torch.all(graph.src[i][1:] >= graph.src[i][:-1])

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
