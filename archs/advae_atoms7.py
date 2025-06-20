from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod, turn_on_actv_size_printing, typsz
from managan.layers_common import *
from managan.config import Config
from managan.tensor_products import TensLinear, TensConv1d, tens_sigmoid, TensSigmoid, TensGroupNorm, TensorRandGen
from managan.codegen_tensor_products import Bee
from managan.jacobi_radenc import radial_encode_8
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResiduesEncodeV2, ResidueAtomEmbed, LinAminoToAtom, LinAtomToAmino
from managan.predictor import ModelState
from managan.learn_common import ErrorTrackingScheduler, get_lr_fn_warmup_and_decay, join_dicts, GenericLossDictTrainer


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor, neighbours_max:int=64):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos, neighbours_max=neighbours_max)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij


def add_tens_prod_submodules(self, dim_a, dim_v, dim_d, chan):
  self.Y_000 = TensLinear(0, chan, dim_a)
  self.Y_110 = TensLinear(0, chan, dim_a)
  self.Y_220 = TensLinear(0, chan, dim_a)
  self.Y_011 = TensLinear(1, chan, dim_v)
  self.Y_101 = TensLinear(1, chan, dim_v)
  self.Y_121 = TensLinear(1, chan, dim_v)
  self.Y_211 = TensLinear(1, chan, dim_v)
  self.Y_022 = TensLinear(2, chan, dim_d)
  self.Y_202 = TensLinear(2, chan, dim_d)
  self.Y_112 = TensLinear(2, chan, dim_d)
  self.Y_222 = TensLinear(2, chan, dim_d)
  self.Y_111 = TensLinear(1, chan, dim_v)
  self.Y_212 = TensLinear(2, chan, dim_d)

@torch.compile(dynamic=True)
def torch_sum_3(X1:torch.Tensor, X2:torch.Tensor, X3:torch.Tensor):
  return X1 + X2 + X3

@torch.compile(dynamic=True)
def torch_sum_5(X1:torch.Tensor, X2:torch.Tensor, X3:torch.Tensor, X4:torch.Tensor, X5:torch.Tensor):
  return X1 + X2 + X3 + X4 + X5

def apply_tens_prod_submodules(self, prods):
  y_000, y_110, y_220, y_011, y_101, y_121, y_211, y_022, y_202, y_112, y_222, y_111, y_212 = prods
  y_0 = torch_sum_3(self.Y_000(y_000), self.Y_110(y_110), self.Y_220(y_220))
  y_1 = torch_sum_5(self.Y_011(y_011), self.Y_101(y_101), self.Y_121(y_121), self.Y_211(y_211), self.Y_111(y_111))
  y_2 = torch_sum_5(self.Y_022(y_022), self.Y_202(y_202), self.Y_112(y_112), self.Y_222(y_222), self.Y_212(y_212))
  return y_0, y_1, y_2

class SelfTensProds(nn.Module):
  def __init__(self, dim_a, dim_v, dim_d, chan, groups=8):
    super().__init__()
    # submodules:
    self.L0 = TensLinear(0, dim_a, chan)
    self.R0 = TensLinear(0, dim_a, chan)
    self.L1 = TensLinear(1, dim_v, chan)
    self.R1 = TensLinear(1, dim_v, chan)
    self.L2 = TensLinear(2, dim_d, chan)
    self.R2 = TensLinear(2, dim_d, chan)
    add_tens_prod_submodules(self, dim_a, dim_v, dim_d, chan)
    # group norms for output
    self.gn_a = TensGroupNorm(0, dim_a, groups)
    self.gn_v = TensGroupNorm(1, dim_v, groups)
    self.gn_d = TensGroupNorm(2, dim_d, groups)
  def forward(self, x_a, x_v, x_d):
    *rest, dim_a = x_a.shape
    *must_be[rest], dim_v, must_be[3] = x_v.shape
    *must_be[rest], dim_d, must_be[3], must_be[3] = x_d.shape
    prod_rest = prod(rest)
    x_a = x_a.reshape(prod_rest, dim_a)
    x_v = x_v.reshape(prod_rest, dim_v, 3)
    x_d = x_d.reshape(prod_rest, dim_d, 3, 3)
    # begin computations
    l_0 = self.L0(x_a)
    r_0 = self.R0(x_a)
    l_1 = self.L1(x_v)
    r_1 = self.R1(x_v)
    l_2 = self.L2(x_d)
    r_2 = self.R2(x_d)
    prods = Bee.apply(l_0, l_1, l_2, r_0, r_1, r_2)
    y_a, y_v, y_d = apply_tens_prod_submodules(self, prods)
    # reshape back to original shape
    y_a = y_a.reshape(*rest, dim_a)
    y_v = y_v.reshape(*rest, dim_v, 3)
    y_d = y_d.reshape(*rest, dim_d, 3, 3)
    return self.gn_a(y_a), self.gn_v(y_v), self.gn_d(y_d)


class MLP(nn.Module):
  def __init__(self, dim_a):
    super().__init__()
    self.lin_direct = nn.Linear(dim_a, dim_a, bias=False)
    self.layers = nn.Sequential(
      nn.Linear(dim_a, 2*dim_a),
      nn.LeakyReLU(0.1),
      nn.Linear(2*dim_a, 2*dim_a),
      nn.LeakyReLU(0.1),
      nn.Linear(2*dim_a, dim_a),
    )
  def forward(self, x):
    return self.lin_direct(x) + self.layers(x)

class VectorMLP(nn.Module):
  def __init__(self, dim_v):
    super().__init__()
    self.lin_direct = TensLinear(1, dim_v, dim_v)
    self.layers = nn.Sequential(
      TensLinear(1, dim_v, 2*dim_v),
      TensSigmoid(1),
      TensLinear(1, 2*dim_v, 2*dim_v),
      TensSigmoid(1),
      TensLinear(1, 2*dim_v, dim_v),
    )
  def forward(self, x):
    return self.lin_direct(x) + self.layers(x)

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

class LocalMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules:
    self.tlm_a = TimeLinearModulation(config["t_embed_hdim"], 0, dim_a)
    self.tlm_v = TimeLinearModulation(config["t_embed_hdim"], 1, dim_v)
    self.tlm_d = TimeLinearModulation(config["t_embed_hdim"], 2, dim_d)
    self.mlp_a = torch.compile(MLP(dim_a))
    self.mlp_v = VectorMLP(dim_v)
    self.lin_d_trans = TensLinear(2, dim_d, dim_d)
  def forward(self, x_a, x_v, x_d, t):
    x_a, x_v, x_d = x_a + self.tlm_a(x_a, t), x_v + self.tlm_v(x_v, t), x_d + self.tlm_d(x_d, t)
    x_a = x_a + self.mlp_a(x_a)
    x_v = x_v + self.mlp_v(x_v)
    x_d = x_d + torch.transpose(self.lin_d_trans(x_d), -1, -2)
    return x_a, x_v, x_d

class PosEmbed(nn.Module):
  """ Embeddings of node-wise relative positions. """
  def __init__(self, dim_v):
    super().__init__()
    self.lin_v = TensLinear(1, 1, dim_v)
  def forward(self, pos_0, pos_1):
    """ pos_0, pos_1: (batch, nodes, 3) """
    pos_0, pos_1 = pos_0[:, :, None], pos_1[:, :, None]
    dpos_v = 0.1*(pos_1 - pos_0)
    return self.lin_v(dpos_v)


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
    return self.lin_l(delta_l) + self.lin_r(delta_r)

class DisplacementTensors(nn.Module):
  """ Given node features and positions, extract a neighbour graph and perform ACE-like embedding. """
  def __init__(self, r0:float, dim_a:int, dim_v:int, dim_d:int):
    super().__init__()
    self.r0 = r0
    self.mlp_radial = nn.Sequential(
      nn.Linear(8, dim_a),
      MLP(dim_a))
    self.readout_v = TensLinear(1, dim_a, dim_v)
    self.readout_d = TensLinear(2, dim_a, dim_d)
  def forward(self, graph, r_ij):
    rad_enc_ij = self.mlp_radial(radial_encode_8(r_ij, self.r0))
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, dim_a)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, dim_a, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, dim_a, 3, 3)
    A_a_i = edges_reduce_src(graph, φ_a_ij) # (batch, nodes, dim_a)
    A_v_i = edges_reduce_src(graph, φ_v_ij) # (batch, nodes, dim_a, 3)
    A_d_i = edges_reduce_src(graph, φ_d_ij) # (batch, nodes, dim_a, 3, 3)
    return A_a_i, self.readout_v(A_v_i), self.readout_d(A_d_i)

class Messages(nn.Module):
  def __init__(self, r0:float, dim_a:int, dim_v:int, dim_d:int, chan:int):
    super().__init__()
    self.r0 = r0
    # submodules
    self.L0 = TensLinear(0, dim_a, chan)
    self.L1 = TensLinear(1, dim_v, chan)
    self.L2 = TensLinear(2, dim_d, chan)
    add_tens_prod_submodules(self, dim_a, dim_v, dim_d, chan)
    self.lin_enc = nn.Linear(8, chan)
    self.mlp_a = torch.compile(MLP(dim_a)) # MLP that we apply to edge activations
  def forward(self, graph, r_ij, x_a, x_v, x_d):
    rad_enc_ij = self.lin_enc(radial_encode_8(r_ij, self.r0))
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, chan)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, chan, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, chan, 3, 3)
    x_a_j = edges_read_dst(graph, self.L0(x_a)) # (edges, chan)
    x_v_j = edges_read_dst(graph, self.L1(x_v)) # (edges, chan, 3)
    x_d_j = edges_read_dst(graph, self.L2(x_d)) # (edges, chan, 3, 3)
    prods = Bee.apply(x_a_j, x_v_j, x_d_j, φ_a_ij, φ_v_ij, φ_d_ij)
    ψ_a_ij, ψ_v_ij, ψ_d_ij = apply_tens_prod_submodules(self, prods)
    ψ_a_ij = ψ_a_ij + self.mlp_a(ψ_a_ij)
    B_a_i = edges_reduce_src(graph, ψ_a_ij) # (batch, nodes, dim_a)
    B_v_i = edges_reduce_src(graph, ψ_v_ij) # (batch, nodes, dim_v, 3)
    B_d_i = edges_reduce_src(graph, ψ_d_ij) # (batch, nodes, dim_d, 3, 3)
    return 0.1*B_a_i, 0.1*B_v_i, 0.1*B_d_i



class AminoChainConv(nn.Module):
  """ The amino acids form a chain we can do 1d convolutions along. """
  def __init__(self, dim_a:int, dim_v:int, dim_d:int):
    super().__init__()
    self.convs_a = nn.Sequential(
      TensConv1d(0, dim_a, 7),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 7),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 7))
    self.convs_v = nn.Sequential(
      TensConv1d(1, dim_v, 7),
      TensConv1d(1, dim_v, 7))
    self.convs_d = nn.Sequential(
      TensConv1d(2, dim_d, 7),
      TensConv1d(2, dim_d, 7))
  def forward(self, x_a, x_v, x_d):
    """ x_a, x_v, x_d: (batch, nodes, dim, (3,)^inds """
    x_a = self.convs_a(x_a)
    x_v = self.convs_v(x_v)
    x_d = self.convs_d(x_d)
    return x_a, x_v, x_d

class SubBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v, dim_d, chan = config["dim_a"], config["dim_v"], config["dim_d"], config["chan"]
    # submodules
    self.lin_readin_a = TensLinear(0, dim_a, dim_a)
    self.lin_readin_v = TensLinear(1, dim_v, dim_v)
    self.lin_readin_d = TensLinear(2, dim_d, dim_d)
    self.disptens_0 = DisplacementTensors(config["r_cut"], dim_a, dim_v, dim_d)
    self.disptens_1 = DisplacementTensors(config["r_cut"], dim_a, dim_v, dim_d)
    self.prods = SelfTensProds(dim_a, dim_v, dim_d, chan)
    self.mlps_out = LocalMLP(config)
  def forward(self, tup):
    # unpack the tuple
    t, contexttup, xtup, pos_0_tup, pos_1_tup = tup
    x_a, x_v, x_d = xtup
    pos_0, graph_0, r_ij_0 = pos_0_tup
    pos_1, graph_1, r_ij_1 = pos_1_tup
    # readin activations
    x_a, x_v, x_d = self.lin_readin_a(x_a), self.lin_readin_v(x_v), self.lin_readin_d(x_d)
    # get displacement tensors
    Δ0_x_a, Δ0_x_v, Δ0_x_d = self.disptens_0(graph_0, r_ij_0)
    Δ1_x_a, Δ1_x_v, Δ1_x_d = self.disptens_1(graph_1, r_ij_1)
    x_a = x_a + Δ0_x_a + Δ1_x_a
    x_v = x_v + Δ0_x_v + Δ1_x_v
    x_d = x_d + Δ0_x_d + Δ1_x_d
    # products
    x_a, x_v, x_d = self.prods(x_a, x_v, x_d)
    # time-embed, MLP, groupnorm
    return self.mlps_out(x_a, x_v, x_d, t)


class AminoSubBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v, dim_d, chan = config["dim_a"], config["dim_v"], config["dim_d"], config["chan"]
    # submodules:
    self.encode_0 = ResiduesEncodeV2(dim_v)
    self.encode_1 = ResiduesEncodeV2(dim_v)
    self.ndisp_embed = NeighbourDispEmbed(dim_v)
    self.pos_embed = PosEmbed(dim_v)
    self.readin_a = LinAtomToAmino(0, dim_a, dim_a)
    self.readin_v = LinAtomToAmino(1, dim_v, dim_v)
    self.readin_d = LinAtomToAmino(2, dim_d, dim_d)
    self.prods = SelfTensProds(dim_a, dim_v, dim_d, chan)
    self.mlps_out = LocalMLP(config)
    self.convs = AminoChainConv(dim_a, dim_v, dim_d)
    self.readout_a = LinAminoToAtom(0, dim_a, dim_a)
    self.readout_v = LinAminoToAtom(1, dim_v, dim_v)
    self.readout_d = LinAminoToAtom(2, dim_d, dim_d)
  def forward(self, tup):
    # unpack the tuple
    t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    box, metadata = contexttup
    x_a, x_v, x_d = xtup # per atom activations
    y_a, y_v, y_d = ytup # per amino activations
    pos_0, graph_0, r_ij_0 = pos_0_tup
    pos_1, graph_1, r_ij_1 = pos_1_tup
    # readin
    y_a = y_a + 0.2*self.readin_a(x_a, metadata)
    y_v = y_v + 0.2*self.readin_v(x_v, metadata)
    y_d = y_d + 0.2*self.readin_d(x_d, metadata)
    # embed
    pos_0_amino, Δ0_y_v = self.encode_0(pos_0, metadata)
    pos_1_amino, Δ1_y_v = self.encode_1(pos_1, metadata)
    y_v = y_v + Δ0_y_v + Δ1_y_v + self.pos_embed(pos_0_amino, pos_1_amino) + self.ndisp_embed(pos_0_amino, pos_1_amino)
    # 1d conv along chain
    Δy_a, Δy_v, Δy_d = self.convs(y_a, y_v, y_d)
    y_a = y_a + Δy_a
    y_v = y_v + Δy_v
    y_d = y_d + Δy_d
    # local computations: product, time-embed, MLP, groupnorm
    Δy_a, Δy_v, Δy_d = self.mlps_out(*self.prods(y_a, y_v, y_d), t)
    y_a = y_a + Δy_a
    y_v = y_v + Δy_v
    y_d = y_d + Δy_d
    # readout
    Δxtup = 0.2*self.readout_a(y_a, metadata), 0.2*self.readout_v(y_v, metadata), 0.2*self.readout_d(y_d, metadata)
    return Δxtup, (y_a, y_v, y_d)


class Block(nn.Module):
  """ Processing block for nets. """
  def __init__(self, config:Config, pos_1_mutable=False):
    super().__init__()
    dim_a, dim_v, dim_d, chan = config["dim_a"], config["dim_v"], config["dim_d"], config["chan"]
    self.pos_1_mutable:bool = pos_1_mutable
    self.r0:float = config["r_cut"] # cutoff radius for atom interactions
    if "neighbours_max" in config:
      self.neighbours_max = config["neighbours_max"]
    else:
      self.neighbours_max = 64
    # submodules:
    self.embed_t = TimeEmbedding(config["t_embed_hdim"], dim_a)
    self.pos_embed = PosEmbed(dim_v)
    self.res_embed = ResidueAtomEmbed(dim_a)
    self.sub_block = SubBlock(config)
    self.messages_0 = Messages(config["r_cut"], dim_a, dim_v, dim_d, chan)
    self.messages_1 = Messages(config["r_cut"], dim_a, dim_v, dim_d, chan)
    self.amino_sub_block = AminoSubBlock(config)
    if pos_1_mutable:
      self.lin_push_pos_1 = TensLinear(1, dim_v, 1)
  def self_init(self):
    if self.pos_1_mutable:
      with torch.no_grad():
        self.lin_push_pos_1.W.zero_()
  def forward(self, tup):
    # unpack the tuple
    t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    box, metadata = contexttup
    x_a, x_v, x_d = xtup # per atom activations
    pos_0, graph_0, r_ij_0 = pos_0_tup
    pos_1, graph_1, r_ij_1 = pos_1_tup
    # graph setup
    if graph_0 is None: # pos_0 was modified since graph was last computed
      graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0, neighbours_max=self.neighbours_max)
      pos_0_tup = pos_0, graph_0, r_ij_0
    if graph_1 is None: # pos_1 was modified since graph was last computed
      graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1, neighbours_max=self.neighbours_max)
      pos_1_tup = pos_1, graph_1, r_ij_1
    # ACE subblock
    Δx_a, Δx_v, Δx_d = self.sub_block((
      t, contexttup, (
        x_a + self.res_embed(metadata) + self.embed_t(t)[:, None],
        x_v + self.pos_embed(pos_0, pos_1),
        x_d),
      pos_0_tup, pos_1_tup))
    x_a = x_a + Δx_a
    x_v = x_v + Δx_v
    x_d = x_d + Δx_d
    # messages
    Δ0_x_a, Δ0_x_v, Δ0_x_d = self.messages_0(graph_0, r_ij_0, x_a, x_v, x_d)
    Δ1_x_a, Δ1_x_v, Δ1_x_d = self.messages_1(graph_1, r_ij_1, x_a, x_v, x_d)
    x_a = x_a + Δ0_x_a + Δ1_x_a
    x_v = x_v + Δ0_x_v + Δ1_x_v
    x_d = x_d + Δ0_x_d + Δ1_x_d
    # amino conv subblock
    Δxtup, ytup = self.amino_sub_block((t, contexttup, (x_a, x_v, x_d), ytup, pos_0_tup, pos_1_tup))
    Δx_a, Δx_v, Δx_d = Δxtup
    x_a = x_a + Δx_a
    x_v = x_v + Δx_v
    x_d = x_d + Δx_d
    # update pos_1
    if self.pos_1_mutable:
      pos_1 = pos_1 + self.lin_push_pos_1(x_v).squeeze(-2)
      pos_1_tup = pos_1, None, None # invalidate previous graph and r_ij
    return t, contexttup, (x_a, x_v, x_d), ytup, pos_0_tup, pos_1_tup


def initialize_tup(self, head, pos_0, pos_1, box, metadata):
  """ Utility function that sets up the tuples that Block operates on.
      self should just be some object containing the correct dim_a, dim_v, dim_d
      values as attributes. head is t or None. """
  device = pos_0.device
  batch,          atoms,          must_be[3] = pos_0.shape
  must_be[batch], must_be[atoms], must_be[3] = pos_1.shape
  aminos = len(metadata.seq)
  pos_0, pos_1 = pos_0.contiguous(), pos_1.contiguous()
  if head is None: # handle this case by just putting zeros
    head = torch.zeros(batch, device=device)
  # build tuple
  contexttup = box, metadata
  xtup = (
    torch.zeros(batch, atoms, self.dim_a, device=device),
    torch.zeros(batch, atoms, self.dim_v, 3, device=device),
    torch.zeros(batch, atoms, self.dim_d, 3, 3, device=device))
  ytup = (
    torch.zeros(batch, aminos, self.dim_a, device=device),
    torch.zeros(batch, aminos, self.dim_v, 3, device=device),
    torch.zeros(batch, aminos, self.dim_d, 3, 3, device=device))
  pos_0_tup = pos_0, None, None
  pos_1_tup = pos_1, None, None
  return head, contexttup, xtup, ytup, pos_0_tup, pos_1_tup

def insert_xs_tup(x_a, x_v, tup):
  """ put x_a, x_v into tup """
  return tup[:2] + ((x_a, x_v) + tup[2][2:],) + tup[3:]


class Decode(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.readin_a = TensLinear(0, config["zdim_a"], self.dim_a)
    self.readin_v = TensLinear(1, config["zdim_v"], self.dim_v)
    self.blocks = nn.Sequential(*[
      Block(config, pos_1_mutable=True)
      for i in range(config["depth"])
    ])
  def forward(self, pos_0, z_a, z_v, box, metadata):
    tup = initialize_tup(self, None, pos_0, pos_0, box, metadata)
    tup = insert_xs_tup(self.readin_a(z_a), self.readin_v(z_v), tup)
    tup = self.blocks(tup)
    t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    return pos_1_tup[0]

class Encode(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, pos_1_mutable=False)
      for i in range(config["depth"])
    ])
    self.readout_a = TensLinear(0, self.dim_a, config["zdim_a"])
    self.readout_v = TensLinear(1, self.dim_v, config["zdim_v"])
  def forward(self, pos_0, pos_1, box, metadata):
    tup = initialize_tup(self, None, pos_0, pos_1, box, metadata)
    tup = self.blocks(tup)
    t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    x_a, x_v, _ = xtup
    return self.readout_a(x_a), self.readout_v(x_v)

class Score(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.readin_a = TensLinear(0, config["zdim_a"], self.dim_a)
    self.readin_v = TensLinear(1, config["zdim_v"], self.dim_v)
    self.blocks = nn.Sequential(*[
      Block(config, pos_1_mutable=True)
      for i in range(config["depth"])
    ])
    self.readout_a = TensLinear(0, self.dim_a, config["zdim_a"])
    self.readout_v = TensLinear(1, self.dim_v, config["zdim_v"])
  def forward(self, t, pos_0, z_a, z_v, box, metadata):
    tup = initialize_tup(self, t, pos_0, pos_0, box, metadata)
    tup = insert_xs_tup(self.readin_a(z_a), self.readin_v(z_v), tup)
    tup = self.blocks(tup)
    t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    x_a, x_v, _ = xtup
    return self.readout_a(x_a), self.readout_v(x_v)


class AdvAe:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.config = config
    self.enc = Encode(config).to(config.device)
    self.dec = Decode(config).to(config.device)
    self.score = Score(config).to(config.device)
    self.enc.apply(weights_init)
    self.dec.apply(weights_init)
    self.score.apply(weights_init)
    assert space_dim(config) == 3
    self.box = config.predictor.get_box()
    self.tensbox = torch.tensor(self.box, dtype=torch.float32, device="cuda")
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.sched_ae = ErrorTrackingScheduler(get_lr_fn_warmup_and_decay(
      self.config["lr_ae"], 0.1, 128, self.config["gamma_ae"]))
    self.optim_ae = torch.optim.AdamW(list(self.enc.parameters()) + list(self.dec.parameters()),
      self.sched_ae.lr(), betas, weight_decay=self.config["weight_decay"])
    self.sched_dn = ErrorTrackingScheduler(get_lr_fn_warmup_and_decay(
      self.config["lr_dn"], 0.1, 128, self.config["gamma_dn"]))
    self.optim_dn = torch.optim.AdamW(list(self.enc.parameters()) + list(self.dec.parameters()),
      self.sched_dn.lr(), betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    ans = AdvAe(config)
    ans.enc.load_state_dict(states["enc"])
    ans.dec.load_state_dict(states["dec"])
    ans.score.load_state_dict(states["score"])
    return ans
  @staticmethod
  def makenew(config):
    return AdvAe(config)
  def save_to_dict(self):
    return {
        "enc": self.enc.state_dict(),
        "dec": self.dec.state_dict(),
        "score": self.score.state_dict(),
      }
  def train_step(self, traj_state):
    """ x: (L, batch, poly_len, 3) """
    x = traj_state.x
    L, batch, atoms, must_be[3] = x.shape
    x_0 = x[:-1].reshape((L - 1)*batch, atoms, 3)
    x_1 = x[1:].reshape((L - 1)*batch, atoms, 3)
    losses = join_dicts(
      self._ae_step(x_0, x_1, traj_state.metadata),
      self._adv_step(x_0, x_1, traj_state.metadata),
      self._score_step(x_0, x_1, traj_state.metadata),
    )
    self.step_count += 1
    self.sched_ae.step(self.step_count, self.optim_ae)
    self.sched_dn.step(self.step_count, self.optim_dn)
    return losses
  def sigma_t(self, t):
    return torch.sin(t*0.5*torch.pi)
  def alpha_t(self, t):
    return torch.cos(t*0.5*torch.pi)
  def _ae_step(self, x_0, x_1, metadata):
    z_a, z_v = self.enc(x_0, x_1, self.box, metadata)
    x_1_pred = self.dec(x_0, z_a, z_v, self.box, metadata)
    loss = ((x_1 - x_1_pred)**2).mean()
    # update weights
    self.optim_ae.zero_grad() # begin ae update
    (self.config["lambda_ae"]*loss).backward()
    return {"loss_ae": loss.item()}
  def _adv_step(self, x_0, x_1, metadata):
    batch, atoms, must_be[3] = x_0.shape
    z_a, z_v = self.enc(x_0, x_1, self.box, metadata)
    t = torch.rand(batch, device=x_0.device)
    sigma_t = self.sigma_t(t)
    alpha_t = self.alpha_t(t)
    epsilon_a = torch.randn_like(z_a)
    epsilon_v = torch.randn_like(z_v)
    z_a_noised = alpha_t[:, None, None]*z_a + sigma_t[:, None, None]*epsilon_a
    z_v_noised = alpha_t[:, None, None, None]*z_v + sigma_t[:, None, None, None]*epsilon_v
    with torch.no_grad():
      score_a, score_v = self.score(t, x_0, z_a_noised, z_v_noised, self.box, metadata)
    # get the true score for a randn-distributed variable
    randn_score_a = -z_a_noised.detach()
    randn_score_v = -z_v_noised.detach()
    d_score_a = randn_score_a - score_a
    d_score_v = randn_score_v - score_v
    loss = -((d_score_a*z_a_noised).mean() + (d_score_v*z_v_noised).mean())
    # update weights
    loss.backward()
    self.optim_ae.step() # end ae update
    return {
      "loss_adv": loss.item(),
      "loss_dsa": (d_score_a**2).mean().item(), "loss_dsv": (d_score_v**2).mean().item()
    }
  def _score_step(self, x_0, x_1, metadata):
    batch,          atoms,          must_be[3] = x_0.shape
    must_be[batch], must_be[atoms], must_be[3] = x_1.shape
    with torch.no_grad():
      z_a, z_v = self.enc(x_0, x_1, self.box, metadata)
      t = torch.rand(batch, device=x_0.device)
      sigma_t = self.sigma_t(t)
      alpha_t = self.alpha_t(t)
      epsilon_a = torch.randn_like(z_a)
      epsilon_v = torch.randn_like(z_v)
      z_a_noised = alpha_t[:, None, None]*z_a + sigma_t[:, None, None]*epsilon_a
      z_v_noised = alpha_t[:, None, None, None]*z_v + sigma_t[:, None, None, None]*epsilon_v
    score_a, score_v = self.score(t, x_0, z_a_noised, z_v_noised, self.box, metadata)
    loss = (
      ((sigma_t[:, None, None]*score_a + epsilon_a)**2).mean((1, 2)) +
      ((sigma_t[:, None, None, None]*score_v + epsilon_v)**2).mean((1, 2, 3))
    ).mean()
    # update weights
    self.optim_dn.zero_grad()
    loss.backward()
    self.optim_dn.step()
    return {"loss_s": loss.item()}
  def generate(self, x_0, metadata, steps=12):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    z_a = torch.randn(batch, atoms, self.config["zdim_a"], device=x_0.device)
    z_v = torch.randn(batch, atoms, self.config["zdim_v"], 3, device=x_0.device)
    ans = self.dec(x_0, z_a, z_v, self.box, metadata)
    return ans.reshape(*leading_dims, atoms, 3)
  def predict(self, state:ModelState):
    with torch.no_grad():
      return self.generate(state.x, state.metadata)



# export model class and trainer class:
modelclass   = AdvAe
trainerclass = GenericLossDictTrainer
