import torch

from config import get_predictor
from utils import must_be, avg_relative_diff


class Symmetry:
  """ base symmetry class corresponding to the trivial symmetry group """
  def scalar(self, x_a):
    """ x_a: (...) """
    return x_a
  def vector(self, x_v):
    """ (..., 3) """
    return x_v
  def matrix(self, x_d):
    """ (..., 3) """
    return x_d
  def pos(self, p_v):
    """ (..., nodes, 3) """
    return p_v
  def apply(self, tens_list, tens_type_list):
    ans = []
    for tens, tens_type in zip(tens_list, tens_type_list):
      if tens_type == "a":
        ans.append(self.scalar(tens))
      elif tens_type == "v":
        ans.append(self.vector(tens))
      elif tens_type == "d":
        ans.append(self.matrix(tens))
      elif tens_type == "p":
        ans.append(self.pos(tens))
      else: assert False, f"symmetry {tens_type} not supported"
    return ans

class RotSymm(Symmetry):
  """ Check that rotating input by a rotation matrix is the same as rotating the output by that matrix. """
  random_rot = torch.tensor([
    [-0.2590, -0.5228,  0.8121],
    [ 0.6468, -0.7184, -0.2562],
    [ 0.7174,  0.4589,  0.5242]], device="cuda")
  def vector(self, x_v):
    return x_v @ self.random_rot
  def matrix(self, x_d):
    return torch.einsum("...ij, ik, jl -> ...kl", x_d, self.random_rot, self.random_rot)
  def pos(self, p_v):
    return p_v @ self.random_rot

class AxisRotSymm(Symmetry):
  """ Like RotSymm, except aligned to axes so that box can be matched. """
  axis_rot = torch.tensor([
    [ 0., -1.,  0.],
    [ 0.,  0., -1.],
    [ 1.,  0.,  0.]], device="cuda")
  def __init__(self, box):
    assert box[0] == box[1] == box[2]
  def vector(self, x_v):
    return x_v @ self.axis_rot
  def matrix(self, x_d):
    assert False, "not implemented!"
  def pos(self, p_v):
    return p_v @ self.axis_rot

class TransSymm(Symmetry):
  """ Check symmetry under uniform translations of the whole system. """
  trans = torch.tensor([10.5, 26., -17.], device="cuda")
  def pos(self, p_v):
    return p_v + self.trans

class BoxSymm(Symmetry):
  """ Check symmetry under discrete translations of different atoms by different box lattice vectors. """
  seed = 0x59AA
  def __init__(self, box):
    self.box = box
  def pos(self, p_v):
    generator = torch.Generator(device=p_v.device)
    generator.manual_seed(self.seed) # ensure displacements will be consistent
    displacements = self.box*torch.randint(-4, 5, p_v.shape, device=p_v.device, generator=generator)
    return p_v + displacements


def main(args):
  if args.test == "modelfile":
    print("Test entire model from file.")
    predictor = get_predictor("model:" + input("Model filename? "))
    box = predictor.get_box()
    if box is None:
      symms = [RotSymm(), TransSymm()]
    else:
      symms = [AxisRotSymm(box), TransSymm(), BoxSymm(box)]
    model = predictor.model
    state = predictor.sample_q(1)
    x = state.x.clone()
    pos_noies, z_a, z_v = model.get_latents(1)
    inputs = [x, pos_noies, z_a, z_v]
    y = model.gen(*inputs)
    for symm in symms:
      inputs_s = symm.apply(inputs, ["p", "v", "a", "v"])
      y_s = symm.pos(y)
      y_s_pred = model.gen(*inputs_s)
      print(symm.__class__.__name__, avg_relative_diff(y_s, y_s_pred))
  elif args.test == "proxattn":
    from attention_layers import ProximityFlashAttentionPeriodic
    from layers_common import VectorSigmoid
    print("Testing periodic version of proximity attention.")
    box_tuple = (16., 16., 16.)
    box = torch.tensor(box_tuple, device="cuda")
    symms = [AxisRotSymm(box), TransSymm(), BoxSymm(box)]
    r0_list = [1., 2., 3., 4., 5.]
    adim, vdim = 16, 16
    proxattn = ProximityFlashAttentionPeriodic(r0_list, adim, vdim, 16, VectorSigmoid)
    proxattn.to("cuda")
    batch = 1
    nodes = 10
    ax = torch.randn(batch, nodes, adim, device="cuda")
    vx = torch.randn(batch, nodes, vdim, 3, device="cuda")
    pos_k = box*torch.rand(batch, len(r0_list), nodes, 3, device="cuda")
    pos_q = box*torch.rand(batch, len(r0_list), nodes, 3, device="cuda")
    ay, vy = proxattn.forward(ax, vx, pos_k, pos_q, box_tuple)
    for symm in symms:
      ax_s, vx_s, pos_k_s, pos_q_s, ay_s, vy_s = symm.apply([ax, vx, pos_k, pos_q, ay, vy], ["a", "v", "p", "p", "a", "v"])
      ay_s_pred, vy_s_pred = proxattn.forward(ax_s, vx_s, pos_k_s, pos_q_s, box_tuple)
      print(symm.__class__.__name__)
      print(avg_relative_diff(ay_s, ay_s_pred))
      print(avg_relative_diff(vy_s, vy_s_pred))
  elif args.test == "graph_embed":
    from archs.wgan_3d_particles import GraphEmbedLayer
    from graph_layers import Graph
    print("Testing graph embed layer...")
    box_tuple = (16., 16., 16.)
    box = torch.tensor(box_tuple, device="cuda")
    symms = [AxisRotSymm(box), TransSymm(), BoxSymm(box)]
    batch = 1
    nodes = 40
    r_cut = 3.
    embed = GraphEmbedLayer(r_cut)
    pos = box*torch.rand(batch, nodes, 3, device="cuda")
    graph = Graph.radius_graph(r_cut, box_tuple, pos)
    y_a, y_v = embed.forward(graph, pos)
    for symm in symms:
      pos_s, y_a_s, y_v_s = symm.apply([pos, y_a, y_v], ["p", "a", "v"])
      graph_s = Graph.radius_graph(r_cut, box_tuple, pos_s)
      y_a_s_pred, y_v_s_pred = embed.forward(graph_s, pos_s)
      print(symm.__class__.__name__)
      print(avg_relative_diff(y_a_s, y_a_s_pred))
      print(avg_relative_diff(y_v_s, y_v_s_pred))
  elif args.test == "tensor_linear":
    from tensor_products import TensLinear
    batch = 4
    nodes = 20
    dim = 32
    symms = [RotSymm()]
    symmap = ["a", "v", "d"]
    for inds in range(3):
      print(f"\n\nINDS: {inds}")
      tens_lin = TensLinear(inds, dim, dim)
      tens_lin.to("cuda")
      x = torch.randn(batch, nodes, dim, *([3]*inds), device="cuda")
      y = tens_lin(x)
      for symm in symms:
        print(symm.__class__.__name__)
        x_s, y_s = symm.apply([x, y], [symmap[inds]]*2)
        y_s_pred = tens_lin(x_s)
        print(avg_relative_diff(y_s, y_s_pred))
  elif args.test == "tensor_products":
    from tensor_products import TensorProds
    batch = 4
    nodes = 40
    dim = 32
    symms = [RotSymm()]
    symmap = ["a", "v", "d"]
    for inds_l in range(3):
      for inds_r in range(3):
        for inds_o in range(3):
          print(f"\n\nL: {inds_l}, R:{inds_r} -> O:{inds_o}")
          try:
            tensprod = TensorProds(inds_l, inds_r, inds_o, dim, dim, dim, 16)
          except AssertionError as e:
            print(e)
            continue
          tensprod.to("cuda")
          x_l = torch.randn(batch, nodes, dim, *([3]*inds_l), device="cuda")
          x_r = torch.randn(batch, nodes, dim, *([3]*inds_r), device="cuda")
          x_o = tensprod(x_l, x_r)
          for symm in symms:
            print(symm.__class__.__name__)
            x_l_s, x_r_s, x_o_s = symm.apply([x_l, x_r, x_o], [symmap[inds_l], symmap[inds_r], symmap[inds_o]])
            x_o_s_pred = tensprod(x_l_s, x_r_s)
            print(avg_relative_diff(x_o_s, x_o_s_pred))



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="symmetry_checker")
  parser.add_argument("test", choices=["modelfile", "proxattn", "graph_embed", "tensor_linear", "tensor_products"])
  main(parser.parse_args())
