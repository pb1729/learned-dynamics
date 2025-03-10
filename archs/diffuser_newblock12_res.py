from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod
from managan.layers_common import *
from managan.config import Config
from managan.tensor_products import *
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResiduesEncode, ResiduesDecode, ResidueEmbed
from managan.predictor import ModelState


def radial_encode_edge(r, n, rmax):
  """ r: (..., 3)
      ans: (..., n)"""
  npi = torch.pi*torch.arange(0, n, device=r.device)
  x_sq = (r**2).sum(-1)/rmax
  return torch.cos(npi*torch.sqrt(x_sq)[..., None])*torch.relu(1. - x_sq)[..., None]


class TimeEmbedding(nn.Module):
  def __init__(self, hdim, outdim):
    super(TimeEmbedding, self).__init__()
    self.hdim = hdim
    self.lin1 = nn.Linear(2*self.hdim, outdim)
  def raw_t_embed(self, t):
    """ t: (batch) """
    ang_freqs = torch.exp(-torch.arange(self.hdim, device=t.device)/(self.hdim - 1))
    phases = t[:, None] * ang_freqs[None, :]
    return torch.cat([
      torch.sin(phases),
      torch.cos(phases),
    ], dim=1)
  def forward(self, t):
    """ t: (batch)
        ans: (batch, outdim) """
    return self.lin1(self.raw_t_embed(t))


class TimeLinearModulation(nn.Module):
  """ https://arxiv.org/pdf/1709.07871
      https://www.sciencedirect.com/science/article/abs/pii/S0031320324001237 """
  def __init__(self, hdim, inds, iodim):
    super().__init__()
    self.inds = inds
    self.embed = TimeEmbedding(hdim, iodim)
    self.lin_i = TensLinear(inds, iodim, iodim)
    self.lin_o = TensLinear(inds, iodim, iodim)
  def forward(self, x, t):
    """ x: (batch, nodes, iodim, [3,]*inds) """
    x = self.lin_i(x)
    x = x * self.embed(t)[:, None, :][(...,) + (None,)*self.inds]
    return self.lin_o(x)


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
    self.tmod_a = TimeLinearModulation(config["t_embed_hdim"], 0, dim_a)
    self.tmod_v = TimeLinearModulation(config["t_embed_hdim"], 1, dim_v)
    self.tmod_d = TimeLinearModulation(config["t_embed_hdim"], 2, dim_d)
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
  def forward(self, t, x_a, x_v, x_d):
    # residual connections
    res_a, res_v, res_d = x_a, x_v, x_d
    # time-dependent feature modulation
    x_a = x_a + self.tmod_a(x_a ,t)
    x_v = x_v + self.tmod_v(x_v ,t)
    x_d = x_d + self.tmod_d(x_d ,t)
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


class Block(nn.Module):
  """ Generic block for generator and discriminator nets. """
  def __init__(self, config:Config):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.r0:float = config["r_cut"] # cutoff radius
    # submodules:
    self.embed_t = TimeEmbedding(config["t_embed_hdim"], dim_a)
    self.arc_embed = ArcEmbed(dim_a)
    self.pos_embed = PosEmbed(dim_a, dim_v)
    self.ndisp_embed = NeighbourDispEmbed(dim_v)
    self.res_embed = ResidueEmbed(dim_a)
    self.sublock_0 = SubBlock(config)
    self.ace_0 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    self.ace_1 = ACEEmbedAVD(self.r0, dim_a, dim_v, dim_d)
    self.sublock_1 = SubBlock(config)
    self.acemes_0 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.acemes_1 = ACEMessageEmbed(self.r0, dim_a, dim_v, dim_d, config["rank"])
    self.subblock_2 = SubBlock(config)
    self.lin_push_0_pos_0 = TensLinear(1, dim_v, 1)
    self.lin_push_0_pos_1 = TensLinear(1, dim_v, 1)
    self.lin_push_1_pos_0 = TensLinear(1, dim_v, 1)
    self.lin_push_1_pos_1 = TensLinear(1, dim_v, 1)
    self.lin_out_a = TensLinear(0, dim_a, dim_a)
    self.lin_out_v = TensLinear(1, dim_v, dim_v)
    self.lin_out_d = TensLinear(2, dim_d, dim_d)
  def self_init(self):
    with torch.no_grad():
      self.lin_push_0_pos_0.W.zero_()
      self.lin_push_0_pos_1.W.zero_()
      self.lin_push_1_pos_0.W.zero_()
      self.lin_push_1_pos_1.W.zero_()
  def forward(self, tup):
    t, pos_0, pos_1, x_a, x_v, x_d, box, metadata = tup
    # residual connections
    res_a, res_v, res_d = x_a, x_v, x_d
    # embed time
    x_a = x_a + self.embed_t(t)[:, None]
    # simple embed positions
    x_a = x_a + self.arc_embed(pos_0.shape[0], pos_0.shape[1], pos_0.device)
    x_v = x_v + self.ndisp_embed(pos_0, pos_1)
    Δx_a, Δx_v = self.pos_embed(pos_0, pos_1)
    x_a, x_v = x_a + Δx_a, x_v + Δx_v
    # embed residue type
    x_a = x_a + self.res_embed(metadata)
    # SUB-BLOCK 0
    x_a, x_v, x_d = self.sublock_0(t, x_a, x_v, x_d)
    # push positions by some vectors
    pos_0 = pos_0 + self.lin_push_0_pos_0(x_v)[:, :, 0]
    pos_1 = pos_1 + self.lin_push_0_pos_1(x_v)[:, :, 0]
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
    x_a, x_v, x_d = self.sublock_1(t, x_a, x_v, x_d)
    # ACE with messages embed positions
    B_a_i_0, B_v_i_0, B_d_i_0 = self.acemes_0(graph_0, r_ij_0, x_a, x_v, x_d)
    B_a_i_1, B_v_i_1, B_d_i_1 = self.acemes_1(graph_1, r_ij_1, x_a, x_v, x_d)
    x_a = x_a + B_a_i_0 + B_a_i_1
    x_v = x_v + B_v_i_0 + B_v_i_1
    x_d = x_d + B_d_i_0 + B_d_i_1
    # SUB-BLOCK 2
    x_a, x_v, x_d = self.subblock_2(t, x_a, x_v, x_d)
    # push positions by some vectors
    pos_0 = pos_0 + self.lin_push_1_pos_0(x_v)[:, :, 0]
    pos_1 = pos_1 + self.lin_push_1_pos_1(x_v)[:, :, 0]
    # linear map to add to residuals
    x_a, x_v, x_d = self.lin_out_a(x_a), self.lin_out_v(x_v), self.lin_out_d(x_d)
    return t, pos_0, pos_1, res_a + x_a, res_v + x_v, res_d + x_d, box, metadata


class Denoiser(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.r0:float = config["r_cut"] # cutoff radius
    self.encode_0 = ResiduesEncode(self.dim_v, nlin=2)
    self.encode_1 = ResiduesEncode(self.dim_v, nlin=2)
    self.decode = ResiduesDecode(self.dim_v, nlin=2)
    self.blocks = nn.Sequential(*[
      Block(config)
      for i in range(config["depth"])
    ])
  def self_init(self):
    self.decode.init_to_zeros()
  def forward(self, t, pos_0, pos_1, box, metadata):
    device = pos_0.device
    pos_0, x_v_0 = self.encode_0(pos_0, metadata)
    pos_1, x_v_1 = self.encode_1(pos_1, metadata)
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = x_v_0 + x_v_1
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    tup = (t, pos_0, pos_1, x_a, x_v, x_d, box, metadata)
    tup = self.blocks(tup)
    t, pos_0, pos_1, x_a, x_v, x_d, *_ = tup
    # decode the output
    return self.decode(pos_1, x_v, metadata)


def nodecay_cosine_schedule(t, sigma_max):
  return torch.cos(0.5*torch.pi*t)/torch.sqrt(sigma_max**-2 + torch.sin(0.5*torch.pi*t)**2)


class DiffusionDenoiser:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.config = config
    self.dn = Denoiser(config).to(config.device)
    self.dn.apply(weights_init)
    assert space_dim(config) == 3
    self.box = config.predictor.get_box()
    self.tensbox = torch.tensor(self.box, dtype=torch.float32, device="cuda")
    self.sigma_max = config["sigma_max"]
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim = torch.optim.AdamW(self.dn.parameters(),
      self.config["lr"], betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    ans = DiffusionDenoiser(config)
    ans.dn.load_state_dict(states["dn"])
    return ans
  @staticmethod
  def makenew(config):
    return DiffusionDenoiser(config)
  def save_to_dict(self):
    return {
        "dn": self.dn.state_dict(),
      }
  def train_step(self, traj_state):
    """ x: (L, batch, poly_len, 3) """
    x = traj_state.x
    L, batch, atoms, must_be[3] = x.shape
    loss = self._diffuser_step(x, traj_state.metadata)
    self.step_count += 1
    if self.step_count % 1024 == 0:
      self.lr_schedule_update()
    return loss
  def lr_schedule_update(self):
    lr_fac = self.config["lr_fac"]
    for group in self.optim.param_groups: # learning rate schedule
      group["lr"] *= lr_fac
  def sigma_t(self, t):
    return nodecay_cosine_schedule(t, self.sigma_max)
  def _diffuser_step(self, x, metadata):
    """ x: (L, batch, poly_len, 3) """
    L, batch, atoms, must_be[3] = x.shape
    x_0 = x[:-1].reshape((L - 1)*batch, atoms, 3)
    x_1 = x[1:].reshape((L - 1)*batch, atoms, 3)
    t = torch.rand((L - 1)*batch, device=x.device)
    epsilon = torch.randn_like(x_0)
    sigma = self.sigma_t(t)[:, None, None]
    x_1_noised = x_1 + sigma*epsilon
    x_1_pred = self.dn(t, x_0, x_1_noised, self.box, metadata)
    loss = (((x_1_pred - x_1)/(0.4 + sigma))**2).mean()
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return loss.item()
  def generate(self, x_0, metadata, steps=12):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    ans = x_0 + self.sigma_t(torch.zeros(1, device=x_0.device))[:, None, None]*self.randgen.randn(1, x_0.shape[:-1])
    t_list = np.linspace(0., 1., steps + 1)
    for i in range(steps):
      t = torch.tensor([t_list[i]], device=x_0.device, dtype=torch.float32)
      tdec= torch.tensor([t_list[i + 1]], device=x_0.device, dtype=torch.float32)
      sigma_t = self.sigma_t(t)[:, None, None]
      sigma_tdec = self.sigma_t(tdec)[:, None, None]
      dsigma = torch.sqrt(sigma_t**2 - sigma_tdec**2)
      pred_noise = ans - self.dn(t, x_0, ans, self.box, metadata)
      ans -= ((dsigma/sigma_t)**2)*pred_noise
      epsilon = torch.randn_like(ans)
      ans += (dsigma*sigma_tdec/sigma_t)*epsilon
    return ans.reshape(*leading_dims, atoms, 3)
  def set_eval(self, bool_eval):
    if bool_eval:
      self.dn.eval()
    else:
      self.dn.train()
  def predict(self, state:ModelState):
    with torch.no_grad():
      return self.generate(state.x, state.metadata)


class DiffusionDenoiserTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    loss = self.model.train_step(trajs)
    print(f"{i}\t ℒᴰ = {loss:05.6f}")
    self.board.scalar("loss", i, loss)


# export model class and trainer class:
modelclass   = DiffusionDenoiser
trainerclass = DiffusionDenoiserTrainer
