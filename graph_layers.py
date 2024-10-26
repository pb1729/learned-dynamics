import torch
import torch.nn as nn

import neighbour_grid_cuda as neighbour_grid

from utils import must_be


class Graph:
  """ Graph class. Supports batch of graphs which have the same number of
      nodes, but may have different connectivity. Graph must be symmetric
      directed graphs not supported. """
  def __init__(self, neighbours, neighbour_counts):
    self.neighbours = neighbours
    self.neighbour_counts = neighbour_counts
  @staticmethod
  def radius_graph(r0, box, R, celllist_max=32, neighbours_max=40):
    """ Return an instance of Graph where nodes are connected if they are within r0 of each other.
        Boundary conditions are periodic, defined by box: tuple(float, float, float).
        R: (batch, nodes, 3) array of node positions """
    neighbours, neighbour_counts = neighbour_grid.get_neighbours(celllist_max, neighbours_max, r0, *box, R)
    neighbours = neighbours[:, :, :neighbour_counts.max()].contiguous()
    ans = Graph(neighbours, neighbour_counts)
    setattr(ans, "box", box) # since graph was created from a periodic box, we should record this
    return ans


class _EdgesRead(torch.autograd.Function):
  @staticmethod
  def forward(ctx, graph:Graph, x) -> torch.Tensor:
    """ x: (batch, nodes, chan) """
    ctx.this_graph_ptr = graph # save a pointer to the graph object
    return neighbour_grid.edges_read(graph.neighbour_counts, graph.neighbours, x)
  @staticmethod
  def backward(ctx, dy):
    """ y: (batch, nodes, neighbours_max, chan) """
    dx = neighbour_grid.edges_reduce(ctx.this_graph_ptr.neighbour_counts, ctx.this_graph_ptr.neighbours, dy)
    return None, dx
edges_read = _EdgesRead.apply

class _EdgesReduce(torch.autograd.Function):
  @staticmethod
  def forward(ctx, graph:Graph, x) -> torch.Tensor:
    """ x: (batch, nodes, neighbours_max, chan) """
    ctx.this_graph_ptr = graph # save a pointer to the graph object
    return neighbour_grid.edges_reduce(graph.neighbour_counts, graph.neighbours, x)
  @staticmethod
  def backward(ctx, dy):
    """ y: (batch, nodes, chan) """
    dx = neighbour_grid.edges_read(ctx.this_graph_ptr.neighbour_counts, ctx.this_graph_ptr.neighbours, dy)
    return None, dx
edges_reduce = _EdgesReduce.apply


# TESTING BY COMPARISON WITH FINITE DIFFERENCE
if __name__ == "__main__":
  batch = 1
  n_particles = 50
  box = (19., 19., 19.)
  tbox = torch.tensor(box, device="cuda")
  R = torch.rand(batch, n_particles, 3, device="cuda")*tbox
  r0 = 6.
  graph = Graph.radius_graph(r0, box, R, neighbours_max=24)
  vals = torch.zeros(batch, n_particles, 4, device="cuda", requires_grad = True)
  def conv(x):
    return edges_reduce(graph, edges_read(graph, x))
  def lossfn(x):
    return conv(x)[0, 0, 0] - conv(x)[0, 1, 1] + 2.*conv(x)[0, 2, 2] - 2.*conv(x)[0, 3, 3]
  loss = lossfn(vals)
  loss.backward()
  delta_vals = 0.01*torch.randn_like(vals, requires_grad=False)
  vals_p = vals + delta_vals
  loss_p = lossfn(vals_p)
  delta_loss = loss_p.item() - loss.item()
  print("should be equal:")
  print(delta_loss, "vs", (vals.grad*delta_vals).sum().item())
