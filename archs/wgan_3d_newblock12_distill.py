from typing_extensions import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod
from managan.layers_common import *
from managan.config import Config, load
from managan.tensor_products import *
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResiduesEncode, ResiduesDecode, ResidueEmbed
from managan.predictor import ModelState


class NormalizeGradient(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    return x
  @staticmethod
  def backward(ctx, dx:torch.Tensor):
    mag = (dx**2).mean(tuple(range(1, len(dx.shape))), keepdim=True)
    return dx/torch.sqrt(1e-10 + mag)
norm_grad = NormalizeGradient.apply

def radial_encode_edge(r, n, rmax):
  """ r: (..., 3)
      ans: (..., n)"""
  npi = torch.pi*torch.arange(0, n, device=r.device)
  x_sq = (r**2).sum(-1)/rmax
  return torch.cos(npi*torch.sqrt(x_sq)[..., None])*torch.relu(1. - x_sq)[..., None]


class PosEmbed(nn.Module):
  """ Embeddings of node-wise relative positions. """
  def __init__(self, dim_a, dim_v):
    super().__init__()
    self.lin_a = TensLinear(0, 1, dim_a)
    self.lin_v = TensLinear(1, 1, dim_v)
  def forward(self, pos_0, pos_1):
    """ pos_0, pos_1: (batch, nodes, 3) """
    pos_0, pos_1 = pos_0[:, :, None], pos_1[:, :, None]
    dpos_v = 0.1*(pos_1 - pos_0)
    return self.lin_a((dpos_v**2).sum(-1)), self.lin_v(dpos_v)


class ArcEmbed(nn.Module):
  """ gets cosine embeddings for node position along the arc of the chain """
  def __init__(self, adim, no_const=True):
    super().__init__()
    self.adim = adim
    self.w_index_offset = float(no_const)
  def forward(self, batch, nodes, device):
    """ ans: (batch, nodes, adim) """
    s = torch.linspace(0., 1., nodes, device=device)[None, :, None].expand(batch, -1, -1)
    w = torch.pi*(self.w_index_offset + torch.arange(self.adim, device=device))[None, None, :]
    return torch.cos(s*w)


class NeighbourDispEmbed(nn.Module):
  """ Embed relative positions between nodes and their neighbours in the chain. """
  def __init__(self, dim_v):
    super().__init__()
    self.lin_l = TensLinear(1, 2, dim_v)
    self.lin_r = TensLinear(1, 2, dim_v)
  def forward(self, pos_0, pos_1):
    """ pos_0, pos_1: (batch, nodes, 3)
        ans: (batch, nodes, dim_v, 3) """
    delta = torch.stack([
      pos_0[:, 1:] - pos_0[:, :-1],
      pos_1[:, 1:] - pos_1[:, :-1],
      ], dim=2) # (batch, nodes - 1, 2, 3)
    delta_l = F.pad(delta, (0,0,  0,0,  0,1)) # (batch, nodes, 2, 3)
    delta_r = F.pad(delta, (0,0,  0,0,  1,0)) # (batch, nodes, 2, 3)
    return 0.1*(self.lin_l(delta_l) + self.lin_r(delta_r))


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij


class ACEEmbedAVD(nn.Module):
  """ Given node features and positions, extract a neighbour graph and perform ACE-like embedding. """
  def __init__(self, r0:float, dim_a:int, dim_v:int, dim_d:int):
    super().__init__()
    self.r0 = r0
    self.lin_a = TensLinear(0, 8, dim_a)
    self.lin_v = TensLinear(1, 8, dim_v)
    self.lin_d = TensLinear(2, 8, dim_d)
  def forward(self, graph, r_ij):
    rad_enc_ij = radial_encode_edge(r_ij, 8, self.r0)
    r_ij = tens_sigmoid(1, r_ij*(17./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, 8)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, 8, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, 8, 3, 3)
    A_a_i = edges_reduce_src(graph, φ_a_ij) # (batch, nodes, 8)
    A_v_i = edges_reduce_src(graph, φ_v_ij) # (batch, nodes, 8, 3)
    A_d_i = edges_reduce_src(graph, φ_d_ij) # (batch, nodes, 8, 3, 3)
    B_a_i = self.lin_a(A_a_i)
    B_v_i = self.lin_v(A_v_i)
    B_d_i = self.lin_d(A_d_i)
    return B_a_i, B_v_i, B_d_i # tensor products done by local ops later...

class ACEMessageEmbed(nn.Module):
  def __init__(self, r0:float, dim_a:int, dim_v:int, dim_d:int, rank:int):
    super().__init__()
    self.r0 = r0
    # φ_a_ij prods
    self.tp_000 = TensorProds(0, 0, 0, dim_a, 8, dim_a, rank)
    self.tp_101 = TensorProds(1, 0, 1, dim_v, 8, dim_v, rank)
    self.tp_202 = TensorProds(2, 0, 2, dim_d, 8, dim_d, rank)
    # φ_v_ij prods
    self.tp_011 = TensorProds(0, 1, 1, dim_a, 8, dim_v, rank)
    self.tp_110 = TensorProds(1, 1, 0, dim_v, 8, dim_a, rank)
    self.tp_112 = TensorProds(1, 1, 2, dim_v, 8, dim_d, rank)
    self.tp_211 = TensorProds(2, 1, 1, dim_d, 8, dim_v, rank)
  def forward(self, graph:Graph, r_ij, x_a, x_v, x_d):
    rad_enc_ij = radial_encode_edge(r_ij, 8, self.r0)
    r_ij = tens_sigmoid(1, r_ij*(17./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                 # (edges, 8)
    φ_v_ij = rad_enc_ij[..., None] * r_ij[..., None, :] # (edges, 8, 3)
    x_a_j = edges_read_dst(graph, x_a) # (edges, dim_a)
    x_v_j = edges_read_dst(graph, x_v) # (edges, dim_v, 3)
    x_d_j = edges_read_dst(graph, x_d) # (edges, dim_d, 3, 3)
    ψ_a_ij = self.tp_000(x_a_j, φ_a_ij) + self.tp_110(x_v_j, φ_v_ij)
    ψ_v_ij = self.tp_011(x_a_j, φ_v_ij) + self.tp_101(x_v_j, φ_a_ij) + self.tp_211(x_d_j, φ_v_ij)
    ψ_d_ij = self.tp_112(x_v_j, φ_v_ij) + self.tp_202(x_d_j, φ_a_ij)
    B_a_i = edges_reduce_src(graph, ψ_a_ij) # (batch, nodes, dim_a)
    B_v_i = edges_reduce_src(graph, ψ_v_ij) # (batch, nodes, dim_v, 3)
    B_d_i = edges_reduce_src(graph, ψ_d_ij) # (batch, nodes, dim_d, 3, 3)
    return B_a_i, B_v_i, B_d_i


class SubBlock(nn.Module):
  """ Submodule of Block """
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules:
    self.linear_mix = AVDFullLinearMix(dim_a, dim_v, dim_d)
    self.tens_prods = AVDFullTensorProds(dim_a, dim_v, dim_d, config["rank"])
    self.gn_a = TensGroupNorm(0, dim_a, config["groups_a"])
    self.gn_v = TensGroupNorm(1, dim_v, config["groups_v"])
    self.gn_d = TensGroupNorm(2, dim_d, config["groups_d"])
    self.convs_a = nn.Sequential(
      TensConv1d(0, dim_a, 7),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 7),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 7))
    self.convs_v = nn.Sequential(
      TensConv1d(1, dim_v, 7),
      TensSigmoid(1),
      TensConv1d(1, dim_v, 7),
      TensSigmoid(1),
      TensConv1d(1, dim_v, 7))
    self.convs_d = nn.Sequential(
      TensConv1d(2, dim_d, 7),
      TensSigmoid(2),
      TensConv1d(2, dim_d, 7),
      TensSigmoid(2),
      TensConv1d(2, dim_d, 7))
    self.mlp_a = nn.Sequential(
      TensLinear(0, dim_a, dim_a),
      nn.LeakyReLU(0.1),
      TensLinear(0, dim_a, dim_a),
      nn.LeakyReLU(0.1),
      TensLinear(0, dim_a, dim_a))
    self.mlp_v = nn.Sequential(
      TensLinear(1, dim_v, dim_v),
      TensSigmoid(1),
      TensLinear(1, dim_v, dim_v),
      TensSigmoid(1),
      TensLinear(1, dim_v, dim_v))
    self.mlp_d = nn.Sequential(
      TensLinear(2, dim_d, dim_d),
      TensSigmoid(2),
      TensLinear(2, dim_d, dim_d),
      TensSigmoid(2),
      TensLinear(2, dim_d, dim_d))
  def forward(self, x_a, x_v, x_d):
    # residual connections
    res_a, res_v, res_d = x_a, x_v, x_d
    # linear mix
    x_a, x_v, x_d = self.linear_mix(x_a, x_v, x_d)
    # tensor products
    Δx_a, Δx_v, Δx_d = self.tens_prods(0.3*x_a, 0.3*x_v, 0.3*x_d) # scale down to prevent from blowing up
    x_a, x_v, x_d = x_a + self.gn_a(Δx_a), x_v + self.gn_v(Δx_v), x_d + self.gn_d(Δx_d)
    # multilayer perceptrons
    return (
      res_a + self.mlp_a(x_a) + self.convs_a(x_a),
      res_v + self.mlp_v(x_v) + self.convs_v(x_v),
      res_d + self.mlp_d(x_d) + self.convs_d(x_d))


class XToEnergy(nn.Module):
  """ Convert activations into a single energy. """
  def __init__(self, dim_a):
    super().__init__()
    self.E_out_lin1_a = TensLinear(0, dim_a, dim_a)
    self.E_out_lin2_a = TensLinear(0, dim_a, 1)
  def forward(self, x_a):
    """ x_a: (batch, nodes, dim_a)
        ans: (batch, nodes) """
    return self.E_out_lin2_a(self.E_out_lin1_a(x_a))[..., 0]


class Block(nn.Module):
  """ Generic block for generator and discriminator nets. """
  def __init__(self, z_scale:float, config:Config, randgen:TensorRandGen, is_disc:bool):
    super().__init__()
    self.randgen = randgen
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.r0:float = config["r_cut"] # cutoff radius
    self.is_disc = is_disc
    # parameters:
    self.z_scale = nn.Parameter(torch.tensor(z_scale))
    # submodules:
    self.arc_embed = ArcEmbed(dim_a)
    self.pos_embed = PosEmbed(dim_a, dim_v)
    self.ndisp_embed = NeighbourDispEmbed(dim_v)
    self.res_embed = ResidueEmbed(dim_a)
    self.sublock_0 = SubBlock(config)
    self.ace_0 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    self.ace_1 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    if not is_disc:
      self.rand_lin_a = TensLinear(0, dim_a, dim_a)
      self.rand_lin_v = TensLinear(1, dim_v, dim_v)
      self.rand_lin_d = TensLinear(2, dim_d, dim_d)
    self.sublock_1 = SubBlock(config)
    self.lin_push_pos_0 = TensLinear(1, dim_v, 1)
    self.acemes_0 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.acemes_1 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.subblock_2 = SubBlock(config)
    self.lin_push_pos_0 = TensLinear(1, dim_v, 1)
    self.lin_push_pos_1 = TensLinear(1, dim_v, 1)
    self.lin_denoise_pos_0 = TensLinear(1, dim_v, 1)
    self.lin_denoise_pos_1 = TensLinear(1, dim_v, 1)
    self.lin_out_a = TensLinear(0, dim_a, dim_a)
    self.lin_out_v = TensLinear(1, dim_v, dim_v)
    self.lin_out_d = TensLinear(2, dim_d, dim_d)
    if is_disc:
      self.x_to_E = XToEnergy(dim_a)
  def self_init(self):
    with torch.no_grad():
      self.lin_push_pos_0.W.zero_()
      self.lin_push_pos_1.W.zero_()
      self.lin_denoise_pos_0.W.zero_()
      self.lin_denoise_pos_1.W.zero_()
  def list_groups(self):
    return (DEFAULT, "slow")
  def grouped_parameters(self, group):
    if group == "slow":
      return (p for p in [self.z_scale])
    else:
      return (p for p in [])
  def forward(self, tup):
    pos_0, pos_1, E, x_a, x_v, x_d, box, metadata = tup
    # residual connections
    res_a, res_v, res_d = x_a, x_v, x_d
    if not self.is_disc:
      # random noising of pos_1
      noise_pos_1 = self.randgen.randn(1, pos_0.shape[:-1])
      pos_1 = pos_1 + self.z_scale*noise_pos_1
      # also add noise to activations
      x_a = x_a + self.rand_lin_a(self.randgen.randn(0, x_a.shape))
      x_v = x_v + self.rand_lin_v(self.randgen.randn(1, x_v.shape[:-1]))
      x_d = x_d + self.rand_lin_d(self.randgen.randn(2, x_d.shape[:-2]))
    # simple embed positions
    x_a = x_a + self.arc_embed(pos_0.shape[0], pos_0.shape[1], pos_0.device)
    x_v = x_v + self.ndisp_embed(pos_0, pos_1)
    Δx_a, Δx_v = self.pos_embed(pos_0, pos_1)
    x_a, x_v = x_a + Δx_a, x_v + Δx_v
    # embed residue type
    x_a = x_a + self.res_embed(metadata)
    # SUB-BLOCK 0
    x_a, x_v, x_d = self.sublock_0(x_a, x_v, x_d)
    # push positions by some vectors
    pos_0 = pos_0 + self.lin_push_pos_0(x_v)[:, :, 0]
    pos_1 = pos_1 + self.lin_push_pos_1(x_v)[:, :, 0]
    # make graphs
    graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0)
    graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1)
    # ACE embed positions
    B_a_i_0, B_v_i_0, B_d_i_0 = self.ace_0(graph_0, r_ij_0)
    B_a_i_1, B_v_i_1, B_d_i_1 = self.ace_1(graph_1, r_ij_1)
    x_a = x_a + B_a_i_0 + B_a_i_1
    x_v = x_v + B_v_i_0 + B_v_i_1
    x_d = x_d + B_d_i_0 + B_d_i_1
    # SUB-BLOCK 1
    x_a, x_v, x_d = self.sublock_1(x_a, x_v, x_d)
    # ACE with messages embed positions
    B_a_i_0, B_v_i_0, B_d_i_0 = self.acemes_0(graph_0, r_ij_0, x_a, x_v, x_d)
    B_a_i_1, B_v_i_1, B_d_i_1 = self.acemes_1(graph_1, r_ij_1, x_a, x_v, x_d)
    x_a = x_a + B_a_i_0 + B_a_i_1
    x_v = x_v + B_v_i_0 + B_v_i_1
    x_d = x_d + B_d_i_0 + B_d_i_1
    # SUB-BLOCK 2
    x_a, x_v, x_d = self.subblock_2(x_a, x_v, x_d)
    # push positions by some vectors
    pos_0 = pos_0 + self.lin_denoise_pos_0(x_v)[:, :, 0]
    pos_1 = pos_1 + self.lin_denoise_pos_1(x_v)[:, :, 0]
    # update energies
    if self.is_disc:
      E = E + self.x_to_E(x_a)
    # linear map to add to residuals
    x_a, x_v, x_d = self.lin_out_a(x_a), self.lin_out_v(x_v), self.lin_out_d(x_d)
    return pos_0, pos_1, E, res_a + x_a, res_v + x_v, res_d + x_d, box, metadata


class DiscHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.x_to_E = XToEnergy(dim_a)
  def forward(self, tup):
    pos_0, pos_1, E, x_a, x_v, x_d, box, metadata = tup
    return (E + self.x_to_E(x_a)).mean(-1) # average over nodes


class Discriminator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.encode_0 = ResiduesEncode(self.dim_v, nlin=2)
    self.encode_1 = ResiduesEncode(self.dim_v, nlin=2)
    self.blocks = nn.Sequential(
      *[
        Block(z_scale, config, randgen, True)
        for z_scale in config["z_scale"]],
      DiscHead(config))
  def forward(self, pos_0, pos_1, box, metadata):
    device = pos_0.device
    pos_0, x_v_0 = self.encode_0(pos_0, metadata)
    pos_1, x_v_1 = self.encode_1(pos_1, metadata)
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = x_v_0 + x_v_1
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    return self.blocks((pos_0, pos_1, 0., x_a, x_v, x_d, box, metadata))


class Generator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.encode_0 = ResiduesEncode(self.dim_v, nlin=2)
    self.decode_2 = ResiduesDecode(self.dim_v, nlin=2)
    self.blocks = nn.Sequential(*[
      Block(z_scale, config, randgen, False)
      for z_scale in config["z_scale"]
    ])
  def self_init(self):
    self.decode_2.init_to_zeros()
  def forward(self, pos_0, box, metadata):
    device = pos_0.device
    pos_0, x_v_0 = self.encode_0(pos_0, metadata)
    batch,          nodes,          must_be[3] = pos_0.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = x_v_0
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    pos_0, pos_1, zero, x_a, x_v, x_d, *_ = self.blocks((pos_0, pos_0, 0., x_a, x_v, x_d, box, metadata))
    # decode the output
    ans = self.decode_2(pos_1, x_v, metadata)
    return norm_grad(ans)


def taxicab(x1, x2, epsilon = 0.01):
  """ x1, x2: (batch, poly_len, 3)
      ans: (batch) """
  return torch.sqrt(((x1 - x2)**2).sum(-1) + epsilon).mean(-1)
def endpoint_penalty(x1, x2, y1, y2):
  """ functional form of endpoint penalty
      x1, x2: (batch, poly_len, 3)
      y1, y2: (batch, heads)
      ans: () """
  # use the taxicab metric (take average distance that nodes moved rather than RMS distance)
  dist = taxicab(x1, x2)[:, None] # (batch, 1)
  # one-sided L1 penalty:
  penalty_l1 = F.relu(torch.abs(y1 - y2)/dist - 1.) # (batch, heads)
  # zero-centered L2 penalty:
  penalty_l2 = 0.2*((y1 - y2)/dist)**2 # (batch, heads)
  return penalty_l1.mean() + penalty_l2.mean()
def get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g, box, metadata):
  """ full computation of endpoint penalty on interpolated data
      x_0: (batch, poly_len, 3)
      x_r, x_g: (batch, poly_len, 3)
      y_r, y_g: (batch, heads)
      ans: () """
  batch, nodes, must_be[3] = x_0.shape
  assert x_r.shape == x_0.shape == x_g.shape
  mix_factors_1 = torch.rand(batch, 1, 1, device=x_0.device)
  mix_factors_2 = torch.rand(batch, 1, 1, device=x_0.device)
  x_1 = mix_factors_1*x_g + (1 - mix_factors_1)*x_r
  x_2 = mix_factors_2*x_g + (1 - mix_factors_2)*x_r
  y_1 = disc(x_0, x_1, box, metadata)
  y_2 = disc(x_0, x_2, box, metadata)
  return (endpoint_penalty(x_r, x_g, y_r, y_g)
        + endpoint_penalty(x_r, x_1, y_r, y_1)
        + endpoint_penalty(x_r, x_2, y_r, y_2)
        + endpoint_penalty(x_g, x_1, y_g, y_1)
        + endpoint_penalty(x_g, x_2, y_g, y_2)
        + endpoint_penalty(x_1, x_2, y_1, y_2))


class WGAN3D:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.config = config
    self.discs = []
    for _ in range(config["ndiscs"]):
      self.add_new_disc()
    self.gen  = Generator(config, self.randgen).to(config.device)
    self.gen.apply(weights_init)
    assert space_dim(config) == 3
    self.box = config.predictor.get_box()
    self.tensbox = torch.tensor(self.box, dtype=torch.float32, device="cuda")
    self.init_optim()
    self.distill_models = None # will be loaded lazily (so only take up memory when training)
  def add_new_disc(self):
    self.discs.append(Discriminator(self.config, self.randgen).to(self.config.device))
    self.discs[-1].apply(weights_init)
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.AdamW(get_params_for_optim(*self.discs, slow={"lr": 0.001*self.config["lr_d"]}),
      self.config["lr_d"], betas, weight_decay=self.config["weight_decay"])
    self.optim_g = torch.optim.AdamW(get_params_for_optim(self.gen, slow={"lr": 0.001*self.config["lr_g"]}),
      self.config["lr_g"], betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    ans = WGAN3D(config)
    for disc, state_dict in zip(ans.discs, states["discs"]):
      disc.load_state_dict(state_dict)
    ans.gen.load_state_dict(states["gen"])
    return ans
  @staticmethod
  def makenew(config):
    return WGAN3D(config)
  def save_to_dict(self):
    return {
        "discs": [disc.state_dict() for disc in self.discs],
        "gen": self.gen.state_dict(),
      }
  def load_diffusers(self):
    ans= {}
    for lagtime in self.config["distill_models"]:
      model = load(self.config["distill_models"][lagtime])
      assert hasattr(model, "dn"), "Expected model to be a denoising model (has nn at .dn)"
      assert hasattr(model, "sigma_t"), "Expected model to be a denoising model (has noise schedule at .sigma_t)"
      # TODO: should we also compare base predictors here?
      assert model.box == self.box, f"box mismatch {model.box} != {self.box}"
      # turn off gradients for model params
      for param in model.dn.parameters():
        param.requires_grad = False
      ans[lagtime] = model
    return ans
  def train_step(self, traj_state):
    """ x: (L, batch, poly_len, 3) """
    x = traj_state.x
    L, batch, atoms, must_be[3] = x.shape
    loss_d = self.discs_step(x, traj_state.metadata)
    if "gen_train" in self.config and not self.config["gen_train"]:
      loss_g = 0.
    else:
      loss_g = self.gen_step(x, traj_state.metadata)
    self.step_count += 1
    if self.step_count % 1024 == 0:
      self.lr_schedule_update()
    return loss_d, loss_g
  def lr_schedule_update(self):
    try:
      lr_d_fac = self.config["lr_d_fac"]
      lr_g_fac = self.config["lr_g_fac"]
    except IndexError:
      lr_d_fac = 0.99
      lr_g_fac = 0.95
    for group in self.optim_g.param_groups: # learning rate schedule
      group["lr"] *= lr_g_fac
    for group in self.optim_d.param_groups: # learning rate schedule
      group["lr"] *= lr_d_fac
  def discs_step(self, x, metadata):
    """ x: (L, batch, poly_len, 3) """
    L, batch, atoms, must_be[3] = x.shape
    x_g = x
    loss = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:-1], metadata)
      x_r = x[nsteps:]
      x_0 = x[:-nsteps]
      loss = loss + self.disc_loss(disc,
        x_0.reshape(batch*(L - nsteps), atoms, 3),
        x_r.reshape(batch*(L - nsteps), atoms, 3),
        x_g.reshape(batch*(L - nsteps), atoms, 3),
        metadata)
    # backprop, update
    self.optim_d.zero_grad()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def disc_loss(self, disc, x_0, x_r, x_g, metadata):
    batch,          atoms,          must_be[3] = x_0.shape
    must_be[batch], must_be[atoms], must_be[3] = x_r.shape
    must_be[batch], must_be[atoms], must_be[3] = x_g.shape
    # train on real data
    y_r = disc(x_0, x_r, self.box, metadata) # (batch, heads)
    # train on generated data
    y_g = disc(x_0, x_g, self.box, metadata) # (batch, heads)
    # endpoint penalty on interpolated data
    endpt_pen = get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g, self.box, metadata)
    # overall loss
    if self.config["hinge"]:
      loss = torch.relu(1. + y_r).mean() + torch.relu(1. - y_g).mean() + self.config["hinge_leak"]*(y_r.mean() - y_g.mean())
    else:
      loss = y_r.mean() - y_g.mean()
    return loss + self.config["lpen_wt"]*endpt_pen
  def gen_step(self, x, metadata):
    if self.distill_models is None:
      self.distill_models = self.load_diffusers()
    L, batch, atoms, must_be[3] = x.shape
    x_g = x
    loss = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:-1], metadata)
      x_0 = x[:-nsteps].reshape((L - nsteps)*batch, atoms, 3)
      x_1_g = x_g.reshape((L - nsteps)*batch, atoms, 3)
      y_g = disc(x_0, x_1_g, self.box, metadata)
      loss = loss + y_g.mean()
      if nsteps in self.distill_models: # add diffuser distillation loss
        model = self.distill_models[nsteps]
        t = torch.rand((L - nsteps)*batch, device=x.device)
        epsilon = torch.randn_like(x_1_g)
        x_1_g_noised = x_1_g + model.sigma_t(t)[:, None, None]*epsilon
        x_1_g_pred = model.dn(t, x_0, x_1_g_noised, self.box, metadata)
        loss = loss + self.config["distill_lambdas"][nsteps]*((x_1_g_pred - x_1_g)**2).mean()
    # backprop, update
    self.optim_g.zero_grad()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def generate(self, x_0, metadata):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    ans = self.gen(x_0, self.box, metadata)
    return ans.reshape(*leading_dims, atoms, 3)
  def set_eval(self, bool_eval):
    if bool_eval:
      for disc in self.discs:
        disc.eval()
      self.gen.eval()
    else:
      for disc in self.discs:
        disc.train()
      self.gen.train()
  def predict(self, state:ModelState):
    with torch.no_grad():
      return self.generate(state.x, state.metadata)


class GANTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    loss_d, loss_g = self.model.train_step(trajs)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}   \t ℒᴳ = {loss_g:05.6f}")
    self.board.scalar("loss_d", i, loss_d)
    self.board.scalar("loss_g", i, loss_g)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer
