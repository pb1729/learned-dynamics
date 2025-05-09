from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod, turn_on_actv_size_printing
from managan.layers_common import *
from managan.config import Config
from managan.tensor_products import TensLinear, TensConv1d, tens_sigmoid, TensSigmoid, TensGroupNorm, TensorRandGen
from managan.codegen_tensor_products import Bee
from managan.jacobi_radenc import radial_encode_8
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResiduesEncodeV2, ResidueAtomEmbed, LinAminoToAtom, LinAtomToAmino
from managan.predictor import ModelState


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos)
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

@torch.compile
def torch_sum_3(X1:torch.Tensor, X2:torch.Tensor, X3:torch.Tensor):
  return X1 + X2 + X3

@torch.compile
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
    delta = 0.1*torch.stack([
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
      graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0)
      pos_0_tup = pos_0, graph_0, r_ij_0
    if graph_1 is None: # pos_1 was modified since graph was last computed
      graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1)
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


class Denoiser(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, pos_1_mutable=True)
      for i in range(config["depth"])
    ])
  def forward(self, t, pos_0, pos_1, box, metadata):
    device = t.device
    batch,          atoms,          must_be[3] = pos_0.shape
    must_be[batch], must_be[atoms], must_be[3] = pos_1.shape
    aminos = len(metadata.seq)
    pos_0, pos_1 = pos_0.contiguous(), pos_1.contiguous()
    # run the main network
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
    tup = t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup
    tup = self.blocks(tup)
    t, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    return pos_1_tup[0]


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
    #turn_on_actv_size_printing(self.dn)
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
