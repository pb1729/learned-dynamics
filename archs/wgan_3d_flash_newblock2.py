from typing_extensions import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from polymer_util import poly_len, space_dim
from utils import must_be, prod
from layers_common import *
from config import Config
from tensor_products import *
from graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from attention_layers import ProbePoints3, ProximityFlashAttentionPeriodic


def radial_encode_edge(r, n, rmax):
  """ r: (..., 3)
      ans: (..., n)"""
  npi = torch.pi*torch.arange(0, n, device=r.device)
  x_sq = (r**2).sum(-1)/rmax
  return torch.cos(npi*torch.sqrt(x_sq)[..., None])*torch.relu(1. - x_sq)[..., None]


class PosEmbed(nn.Module):
  """ Embeddings of node-wise relative positions. """
  def __init__(self, dim):
    super().__init__()
    self.lin = VecLinear(3, dim)
    self.output_gn = TensGroupNorm(1, dim, 1)
  def forward(self, pos_0, pos_1, pos_2):
    """ pos_0, pos_1, pos_2: (batch, nodes, 3) """
    pos_0, pos_1, pos_2 = pos_0[:, :, None], pos_1[:, :, None], pos_2[:, :, None]
    return self.output_gn(self.lin(
      torch.cat([
        pos_1 - pos_0,
        pos_2 - pos_1,
        pos_0 - pos_2], dim=2)))

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


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src) # nested(batch, (edges, 3))
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
    # φ_d_ij prods
    self.tp_022 = TensorProds(0, 2, 2, dim_a, 8, dim_d, rank)
    self.tp_121 = TensorProds(1, 2, 1, dim_v, 8, dim_v, rank)
    self.tp_220 = TensorProds(2, 2, 0, dim_d, 8, dim_a, rank)
    self.tp_222 = TensorProds(2, 2, 2, dim_d, 8, dim_d, rank)
  def forward(self, graph, r_ij, x_a, x_v, x_d):
    rad_enc_ij = radial_encode_edge(r_ij, 8, self.r0)
    r_ij = tens_sigmoid(1, r_ij*(17./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, 8)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, 8, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, 8, 3, 3)
    x_a_j = edges_read_dst(graph, x_a) # nested(batch, (edges, dim_a))
    x_v_j = edges_read_dst(graph, x_v) # nested(batch, (edges, dim_v, 3))
    x_d_j = edges_read_dst(graph, x_d) # nested(batch, (edges, dim_d, 3, 3))
    ψ_a_ij = self.tp_000(x_a_j, φ_a_ij) + self.tp_110(x_v_j, φ_v_ij) + self.tp_220(x_d_j, φ_d_ij)
    ψ_v_ij = self.tp_011(x_a_j, φ_v_ij) + self.tp_101(x_v_j, φ_a_ij) + self.tp_121(x_v_j, φ_d_ij) + self.tp_211(x_d_j, φ_v_ij)
    ψ_d_ij = self.tp_022(x_a_j, φ_d_ij) + self.tp_112(x_v_j, φ_v_ij) + self.tp_202(x_d_j, φ_a_ij) + self.tp_222(x_d_j, φ_d_ij)
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
    self.mlp_a = nn.Sequential(
      TensConv1d(0, dim_a, 7),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 7),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 7))
    self.mlp_v = nn.Sequential(
      TensConv1d(1, dim_v, 7),
      TensSigmoid(1),
      TensConv1d(1, dim_v, 7),
      TensSigmoid(1),
      TensConv1d(1, dim_v, 7))
    self.mlp_d = nn.Sequential(
      TensConv1d(2, dim_d, 7),
      TensSigmoid(2),
      TensConv1d(2, dim_d, 7),
      TensSigmoid(2),
      TensConv1d(2, dim_d, 7))
    self.gn_a = TensGroupNorm(0, dim_a, config["groups_a"])
    self.gn_v = TensGroupNorm(1, dim_v, config["groups_v"])
    self.gn_d = TensGroupNorm(2, dim_d, config["groups_d"])
  def forward(self, x_a, x_v, x_d):
    # residual connections
    res_a, res_v, res_d = x_a, x_v, x_d
    # linear mix
    x_a, x_v, x_d = self.linear_mix(x_a, x_v, x_d)
    # tensor products
    Δx_a, Δx_v, Δx_d = self.tens_prods(x_a, x_v, x_d)
    x_a, x_v, x_d = x_a + Δx_a, x_v + Δx_v, x_d + Δx_d
    # multilayer perceptrons
    x_a = x_a + self.mlp_a(x_a)
    x_v = x_v + self.mlp_v(x_v)
    x_d = x_d + self.mlp_d(x_d)
    # apply group norms
    x_a, x_v, x_d = self.gn_a(x_a), self.gn_v(x_v), self.gn_d(x_d)
    return res_a + x_a, res_v + x_v, res_d + x_d

class Block(nn.Module):
  """ Generic block for generator and discriminator nets. """
  def __init__(self, z_scale:float, config:Config, randgen:TensorRandGen):
    super().__init__()
    self.z_scale = z_scale
    self.randgen = randgen
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.r0:float = config["r_cut"] # cutoff radius
    # submodules:
    self.arc_embed = ArcEmbed(dim_a)
    self.pos_embed = PosEmbed(dim_v)
    self.ace_0 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    self.ace_1 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    self.ace_2 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    self.rand_lin_a = TensLinear(0, dim_a, dim_a)
    self.rand_lin_v = TensLinear(1, dim_v, dim_v)
    self.rand_lin_d = TensLinear(2, dim_d, dim_d)
    self.sublock_0 = SubBlock(config)
    self.acemes_0 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.acemes_1 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.acemes_2 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.sublock_1 = SubBlock(config)
    self.lin_push_pos = TensLinear(1, dim_v, 1)
    self.probe_pts_q = ProbePoints3(len(config["r0_list"]), dim_v)
    self.probe_pts_k = ProbePoints3(len(config["r0_list"]), dim_v)
    self.prox_attn = ProximityFlashAttentionPeriodic(config["r0_list"], dim_a, dim_v, 16, vec_actv_class=VectorSigmoid)
    self.gn_a = TensGroupNorm(0, dim_a, config["groups_a"])
    self.gn_v = TensGroupNorm(1, dim_v, config["groups_v"])
    self.gn_d = TensGroupNorm(2, dim_d, config["groups_d"])
  def forward(self, tup):
    pos_0, pos_1, pos_2, x_a, x_v, x_d, box = tup
    # residual connections
    res_a, res_v, res_d = x_a, x_v, x_d
    # random noising of pos_2
    pos_2 = pos_2 + self.z_scale*self.randgen.randn(1, pos_0.shape[:-1])
    # simple embed positions
    x_a = x_a + self.arc_embed(pos_0.shape[0], pos_0.shape[1], pos_0.device)
    x_v = x_v + self.pos_embed(pos_0, pos_1, pos_2)
    # make graphs
    graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0)
    graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1)
    graph_2, r_ij_2 = graph_setup(self.r0, box, pos_2)
    # ACE embed positions
    B_a_i_0, B_v_i_0, B_d_i_0 = self.ace_0(graph_0, r_ij_0)
    B_a_i_1, B_v_i_1, B_d_i_1 = self.ace_1(graph_1, r_ij_1)
    B_a_i_2, B_v_i_2, B_d_i_2 = self.ace_2(graph_2, r_ij_2)
    x_a = x_a + B_a_i_0 + B_a_i_1 + B_a_i_2
    x_v = x_v + B_v_i_0 + B_v_i_1 + B_v_i_2
    x_d = x_d + B_d_i_0 + B_d_i_1 + B_d_i_2
    # add noise
    x_a = x_a + self.rand_lin_a(self.randgen.randn(0, x_a.shape))
    x_v = x_v + self.rand_lin_v(self.randgen.randn(1, x_v.shape[:-1]))
    x_d = x_d + self.rand_lin_d(self.randgen.randn(2, x_d.shape[:-2]))
    # SUB-BLOCK 0
    x_a, x_v, x_d = self.sublock_0(x_a, x_v, x_d)
    # ACE with messages embed positions
    B_a_i_0, B_v_i_0, B_d_i_0 = self.acemes_0(graph_0, r_ij_0, x_a, x_v, x_d)
    B_a_i_1, B_v_i_1, B_d_i_1 = self.acemes_1(graph_1, r_ij_1, x_a, x_v, x_d)
    B_a_i_2, B_v_i_2, B_d_i_2 = self.acemes_2(graph_2, r_ij_2, x_a, x_v, x_d)
    x_a = x_a + B_a_i_0 + B_a_i_1 + B_a_i_2
    x_v = x_v + B_v_i_0 + B_v_i_1 + B_v_i_2
    x_d = x_d + B_d_i_0 + B_d_i_1 + B_d_i_2
    # SUB-BLOCK 1
    x_a, x_v, x_d = self.sublock_1(x_a, x_v, x_d)
    # push pos_2 by some vectors
    pos_2 = pos_2 + self.lin_push_pos(x_v)[:, :, 0]
    # flash attention
    probes_q = self.probe_pts_q(x_v, pos_2)
    probes_k = self.probe_pts_k(x_v, pos_2)
    Δx_a, Δx_v = self.prox_attn(x_a, x_v, probes_k, probes_q, box)
    x_a, x_v = x_a + Δx_a, x_v + Δx_v
    # group norm
    x_a, x_v, x_d = self.gn_a(x_a), self.gn_v(x_v), self.gn_d(x_d)
    return pos_0, pos_1, pos_2, res_a + x_a, res_v + x_v, res_d + x_d, box


class Discriminator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.blocks = nn.Sequential(*[
      Block(z_scale, config, randgen)
      for z_scale in config["z_scale"]
    ])
    self.lin_a = nn.Linear(self.dim_a, config["heads"], bias=False)
    self.lin_v = VecLinear(self.dim_v, config["heads"])
    self.lin_d = TensTrace(2, self.dim_d, config["heads"])
  def forward(self, pos_0, pos_1, box):
    device = pos_0.device
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = torch.zeros(batch, nodes, self.dim_v, 3, device=device)
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    pos_0, pos_1, pos_2, x_a, x_v, x_d, _ = self.blocks((pos_0, pos_1, pos_0, x_a, x_v, x_d, box))
    # use taxicab distance of pos_1 and pos_2 as part of discriminator output
    y = taxicab(pos_1, pos_2)[:, None] # (batch, 1)
    # activation-based output, with distinct heads
    y = y + self.lin_a(x_a.mean(1))                           # (batch, heads)
    y = y + torch.sqrt((self.lin_v(x_v.mean(1))**2).sum(-1))  # (batch, heads)
    y = y + self.lin_d(x_d.mean(1))                           # (batch, heads)
    return y


class Generator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.blocks = nn.Sequential(*[
      Block(z_scale, config, randgen)
      for z_scale in config["z_scale"]
    ])
  def forward(self, pos_0, box):
    device = pos_0.device
    batch,          nodes,          must_be[3] = pos_0.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = torch.zeros(batch, nodes, self.dim_v, 3, device=device)
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    pos_0, pos_1, pos_2, x_a, x_v, x_d, _ = self.blocks((pos_0, pos_0, pos_0, x_a, x_v, x_d, box))
    return pos_2


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
def get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g, box):
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
  y_1 = disc(x_0, x_1, box)
  y_2 = disc(x_0, x_2, box)
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
    self.discs = [Discriminator(config, self.randgen).to(config.device) for _ in range(config["ndiscs"])]
    self.gen  = Generator(config, self.randgen).to(config.device)
    self.config = config
    assert space_dim(config) == 3
    self.n_nodes = poly_len(config)
    self.tensbox = config.predictor.get_box()
    self.box = (self.tensbox[0].item(), self.tensbox[1].item(), self.tensbox[2].item())
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.AdamW((param for disc in self.discs for param in disc.parameters()),
      self.config["lr_d"], betas, weight_decay=self.config["weight_decay"])
    self.optim_g = torch.optim.AdamW(self.gen.parameters(),
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
    ans = WGAN3D(config)
    for disc in ans.discs:
      disc.apply(weights_init)
    ans.gen.apply(weights_init)
    return ans
  def save_to_dict(self):
    return {
        "discs": [disc.state_dict() for disc in self.discs],
        "gen": self.gen.state_dict(),
      }
  def train_step(self, x):
    """ x: (batch, L, poly_len, 3) """
    batch, L, must_be[self.n_nodes], must_be[3] = x.shape
    loss_d = self.discs_step(x)
    loss_g = self.gen_step(x)
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
  def discs_step(self, x):
    """ x: (batch, L, poly_len, 3) """
    batch, L, must_be[self.n_nodes], must_be[3] = x.shape
    x_g = x
    loss = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:, :-1])
      x_r = x[:, nsteps:]
      x_0 = x[:, :-nsteps]
      loss = loss + self.disc_loss(disc,
        x_0.reshape(batch*(L - nsteps), self.n_nodes, 3),
        x_r.reshape(batch*(L - nsteps), self.n_nodes, 3),
        x_g.reshape(batch*(L - nsteps), self.n_nodes, 3))
    # backprop, update
    self.optim_d.zero_grad()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def disc_loss(self, disc, x_0, x_r, x_g):
    batch,          must_be[self.n_nodes], must_be[3] = x_0.shape
    must_be[batch], must_be[self.n_nodes], must_be[3] = x_r.shape
    must_be[batch], must_be[self.n_nodes], must_be[3] = x_g.shape
    # train on real data
    y_r = disc(x_0, x_r, self.box) # (batch, heads)
    # train on generated data
    y_g = disc(x_0, x_g, self.box) # (batch, heads)
    # endpoint penalty on interpolated data
    endpt_pen = get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g, self.box)
    # overall loss
    if self.config["hinge"]:
      loss = torch.relu(1. + y_r).mean() + torch.relu(1. - y_g).mean() + self.config["hinge_leak"]*(y_r.mean() - y_g.mean())
    else:
      loss = y_r.mean() - y_g.mean()
    return loss + self.config["lpen_wt"]*endpt_pen
  def gen_step(self, x):
    x_g = x
    loss = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:, :-1])
      x_0 = x[:, :-nsteps]
      y_g = disc(x_0.reshape(-1, self.n_nodes, 3), x_g.reshape(-1, self.n_nodes, 3), self.box)
      loss = loss + y_g.mean()
    # backprop, update
    self.optim_g.zero_grad()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def generate(self, x_0):
    *leading_dims, must_be[self.n_nodes], must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, self.n_nodes, 3)
    ans = self.gen(x_0, self.box)
    return ans.reshape(*leading_dims, self.n_nodes, 3)
  def set_eval(self, bool_eval):
    if bool_eval:
      for disc in self.discs:
        disc.eval()
      self.gen.eval()
    else:
      for disc in self.discs:
        disc.train()
      self.gen.train()
  def predict(self, cond):
    with torch.no_grad():
      return self.generate(cond)


class GANTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    config = self.model.config
    loss_d, loss_g = self.model.train_step(trajs)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}   \t ℒᴳ = {loss_g:05.6f}")
    self.board.scalar("loss_d", i, loss_d)
    self.board.scalar("loss_g", i, loss_g)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer
