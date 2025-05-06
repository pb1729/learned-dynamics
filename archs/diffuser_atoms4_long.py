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
from managan.codegen_tensor_products import Ant16
from managan.jacobi_radenc import radial_encode_8
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResiduesEncodeV2, ResidueAtomEmbed, LinAminoToAtom, LinAtomToAmino
from managan.predictor import ModelState


# the following shape manipulation functions are for dealing with long positions
def to_high_pos(pos, L):
  batch, atoms, must_be[L], must_be[3] = pos.shape
  return torch.permute(pos, (2, 0, 1, 3))

def to_low_pos(pos, L):
  must_be[L], batch, atoms, must_be[3] = pos.shape
  return torch.permute(pos, (1, 2, 0, 3))

def fuse_high_pos(pos, L):
  must_be[L], batch, atoms, must_be[3] = pos.shape
  return pos.reshape(L*batch, atoms, 3)

def unfuse_high_pos(pos, L):
  L_batch, atoms, must_be[3] = pos.shape
  assert L_batch % L == 0, "first dim not a multiple of L"
  return pos.reshape(L, L_batch//L, atoms, 3)

def to_low_actv(inds, x, L):
  must_be[L], batch, atoms, chan, *must_be[(3,)*inds] = x.shape
  return torch.permute(x, (1, 2, 0, 3) + tuple(range(4, 4+inds)))

def to_high_actv(inds, x, L):
  batch, atoms, must_be[L], chan, *must_be[(3,)*inds] = x.shape
  return torch.permute(x, (2, 0, 1, 3) + tuple(range(4, 4+inds)))

def fuse_high_actv(inds, x, L):
  must_be[L], batch, atoms, chan, *must_be[(3,)*inds] = x.shape
  return x.reshape(L*batch, atoms, chan, *(3,)*inds)

def unfuse_high_actv(inds, x, L):
  L_batch, atoms, chan, *must_be[(3,)*inds] = x.shape
  assert L_batch % L == 0, "first dim not a multiple of L"
  return x.reshape(L, L_batch//L, atoms, chan, *(3,)*inds)

def fuse_low_actv(inds, x, L):
  batch, atoms, must_be[L], chan, *must_be[(3,)*inds] = x.shape
  return x.reshape(batch, atoms, L*chan, *(3,)*inds)

def unfuse_low_actv(inds, x, L):
  batch, atoms, L_chan, *must_be[(3,)*inds] = x.shape
  assert L_chan % L == 0, "chan dim not a multiple of L"
  return x.reshape(batch, atoms, L, L_chan//L, *(3,)*inds)


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor):
  """ pos: (L, batch, atoms, 3) -- must be in high position """
  pos = fuse_high_pos(pos, pos.shape[0])
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij


def add_tens_prod_parameters(self, dim_a, dim_v, dim_d, dim_l):
  """ MUTATES: self """
  nfac = (dim_l*max(dim_a, dim_v, dim_d))**-0.5
  self.P_000 = nn.Parameter(torch.randn(dim_a, dim_l, dim_a)*nfac)
  self.P_110 = nn.Parameter(torch.randn(dim_a, dim_l, dim_v)*nfac)
  self.P_220 = nn.Parameter(torch.randn(dim_a, dim_l, dim_d)*nfac)
  self.P_011 = nn.Parameter(torch.randn(dim_v, dim_l, dim_v)*nfac)
  self.P_101 = nn.Parameter(torch.randn(dim_v, dim_l, dim_a)*nfac)
  self.P_121 = nn.Parameter(torch.randn(dim_v, dim_l, dim_d)*nfac)
  self.P_211 = nn.Parameter(torch.randn(dim_v, dim_l, dim_v)*nfac)
  self.P_111 = nn.Parameter(torch.randn(dim_v, dim_l, dim_v)*nfac)
  self.P_022 = nn.Parameter(torch.randn(dim_d, dim_l, dim_d)*nfac)
  self.P_112 = nn.Parameter(torch.randn(dim_d, dim_l, dim_v)*nfac)
  self.P_202 = nn.Parameter(torch.randn(dim_d, dim_l, dim_a)*nfac)
  self.P_222 = nn.Parameter(torch.randn(dim_d, dim_l, dim_d)*nfac)
  self.P_212 = nn.Parameter(torch.randn(dim_d, dim_l, dim_v)*nfac)

class SelfTensProds(nn.Module):
  def __init__(self, dim_a, dim_v, dim_d, groups=8):
    super().__init__()
    # parameters
    add_tens_prod_parameters(self, 64, 48, 32, 16)
    # submodules:
    self.lin_a_in = TensLinear(0, dim_a, 64)
    self.lin_a_out = TensLinear(0, 64, dim_a)
    self.lin_v_in = TensLinear(1, dim_v, 48)
    self.lin_v_out = TensLinear(1, 48, dim_v)
    self.lin_d_in = TensLinear(2, dim_d, 32)
    self.lin_d_out = TensLinear(2, 32, dim_d)
    # submodules for left side of products:
    self.lin_left_000 = TensLinear(0, dim_a, 16)
    self.lin_left_110 = TensLinear(1, dim_v, 16)
    self.lin_left_220 = TensLinear(2, dim_d, 16)
    self.lin_left_011 = TensLinear(0, dim_a, 16)
    self.lin_left_101 = TensLinear(1, dim_v, 16)
    self.lin_left_121 = TensLinear(1, dim_v, 16)
    self.lin_left_211 = TensLinear(2, dim_d, 16)
    self.lin_left_111 = TensLinear(1, dim_v, 16)
    self.lin_left_022 = TensLinear(0, dim_a, 16)
    self.lin_left_112 = TensLinear(1, dim_v, 16)
    self.lin_left_202 = TensLinear(2, dim_d, 16)
    self.lin_left_222 = TensLinear(2, dim_d, 16)
    self.lin_left_212 = TensLinear(2, dim_d, 16)
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
    left_000 = self.lin_left_000(x_a)
    left_110 = self.lin_left_110(x_v)
    left_220 = self.lin_left_220(x_d)
    left_011 = self.lin_left_011(x_a)
    left_101 = self.lin_left_101(x_v)
    left_121 = self.lin_left_121(x_v)
    left_211 = self.lin_left_211(x_d)
    left_111 = self.lin_left_111(x_v)
    left_022 = self.lin_left_022(x_a)
    left_112 = self.lin_left_112(x_v)
    left_202 = self.lin_left_202(x_d)
    left_222 = self.lin_left_222(x_d)
    left_212 = self.lin_left_212(x_d)
    # Convert inputs
    x_a_mid = self.lin_a_in(x_a)
    x_v_mid = self.lin_v_in(x_v)
    x_d_mid = self.lin_d_in(x_d)
    y_a_mid, y_v_mid, y_d_mid = Ant16.apply(
      x_a_mid, x_v_mid, x_d_mid,
      self.P_000, left_000, self.P_110, left_110, self.P_220, left_220,
      self.P_011, left_011, self.P_101, left_101, self.P_121, left_121, self.P_211, left_211, self.P_111, left_111,
      self.P_022, left_022, self.P_202, left_202, self.P_112, left_112, self.P_222, left_222, self.P_212, left_212
    )
    # residual connection
    y_a = self.lin_a_out(y_a_mid) + x_a
    y_v = self.lin_v_out(y_v_mid) + x_v
    y_d = self.lin_d_out(y_d_mid) + x_d
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
  def __init__(self, L_0, L_1, dim_v):
    super().__init__()
    self.L_0 = L_0
    self.L_1 = L_1
    self.L = L_0 + L_1
    self.lin_v = TensLinear(1, self.L*self.L, dim_v)
  def forward(self, pos_0, pos_1):
    """ pos_i: (L_i, batch, nodes, 3) -- must be in high position
        ans: (batch, nodes, dim_v, 3) """
    must_be[self.L_0], batch, nodes, must_be[3] = pos_0.shape
    *must_be[self.L_1, batch, nodes], must_be[3] = pos_1.shape
    pos = torch.cat([pos_0, pos_1], dim=0)
    pos_lp = to_low_pos(pos, self.L)
    dpos_v = 0.1*(pos_lp[:, :, None, :] - pos_lp[:, :, :, None]).reshape(batch, nodes, self.L*self.L, 3)
    return self.lin_v(dpos_v)

class NeighbourDispEmbed(nn.Module):
  """ Embed relative positions between nodes and their neighbours in the chain. """
  def __init__(self, L_0, L_1, dim_v):
    super().__init__()
    self.L_0 = L_0
    self.L_1 = L_1
    self.L = L_0 + L_1
    self.lin_l = TensLinear(1, self.L, dim_v)
    self.lin_r = TensLinear(1, self.L, dim_v)
  def forward(self, pos_0, pos_1):
    """ pos_i: (L_i, batch, nodes, 3) -- must be in high position
        ans: (batch, nodes, dim_v, 3) """
    must_be[self.L_0], batch, nodes, must_be[3] = pos_0.shape
    *must_be[self.L_1, batch, nodes], must_be[3] = pos_1.shape
    pos = torch.cat([pos_0, pos_1], dim=0)
    pos_lp = to_low_pos(pos, self.L)
    delta = pos_lp[:, 1:] - pos_lp[:, :-1]    # (batch, nodes - 1, L, 3)
    delta_l = F.pad(delta, (0,0,  0,0,  0,1)) # (batch, nodes, L, 3)
    delta_r = F.pad(delta, (0,0,  0,0,  1,0)) # (batch, nodes, L, 3)
    return (self.lin_l(delta_l) + self.lin_r(delta_r))

class DisplacementTensors(nn.Module):
  """ Given node features and positions, perform ACE-like embedding. """
  def __init__(self, r0:float, dim:int, dim_a:int, dim_v:int, dim_d:int, L:int):
    super().__init__()
    self.r0 = r0
    self.L = L
    self.mlp_radial = nn.Sequential(
      nn.Linear(8, dim),
      MLP(dim))
    self.readout_a = TensLinear(0, L*dim, dim_a)
    self.readout_v = TensLinear(1, L*dim, dim_v)
    self.readout_d = TensLinear(2, L*dim, dim_d)
  def forward(self, graph, r_ij):
    rad_enc_ij = self.mlp_radial(radial_encode_8(r_ij, self.r0))
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, dim)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, dim, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, dim, 3, 3)
    A_a_i = edges_reduce_src(graph, φ_a_ij) # (L*batch, nodes, dim)
    A_v_i = edges_reduce_src(graph, φ_v_ij) # (L*batch, nodes, dim, 3)
    A_d_i = edges_reduce_src(graph, φ_d_ij) # (L*batch, nodes, dim, 3, 3)
    A_a_i = fuse_low_actv(0, to_low_actv(0, unfuse_high_actv(0, A_a_i, self.L), self.L), self.L) # (batch, nodes, L*dim)
    A_v_i = fuse_low_actv(1, to_low_actv(1, unfuse_high_actv(1, A_v_i, self.L), self.L), self.L) # (batch, nodes, L*dim, 3)
    A_d_i = fuse_low_actv(2, to_low_actv(2, unfuse_high_actv(2, A_d_i, self.L), self.L), self.L) # (batch, nodes, L*dim, 3, 3)
    return self.readout_a(A_a_i), self.readout_v(A_v_i), self.readout_d(A_d_i)

class Messages(nn.Module):
  def __init__(self, r0:float, dim_a:int, dim_v:int, dim_d:int, L:int):
    super().__init__()
    self.r0 = r0
    self.L = L
    # parameters:
    add_tens_prod_parameters(self, dim_a, dim_v, dim_d, 8)
    # submodules
    self.readin_a = TensLinear(0, dim_a, 128*L)
    self.readin_v = TensLinear(1, dim_v, 96*L)
    self.readin_d = TensLinear(2, dim_d, 64*L)
    self.readout_a = TensLinear(0, 128*L, dim_a)
    self.readout_v = TensLinear(1, 96*L, dim_v)
    self.readout_d = TensLinear(2, 64*L, dim_d)
    self.mlp_a = torch.compile(MLP(dim_a)) # MLP that we apply to edge activations
    self.mlp_v = VectorMLP(dim_v)
  def forward(self, graph, r_ij, x_a, x_v, x_d):
    rad_enc_ij = radial_encode_8(r_ij, self.r0)
    x_a = fuse_high_actv(0, to_high_actv(0, unfuse_low_actv(0, self.readin_a(x_a), self.L), self.L), self.L) # (L*batch, nodes, 128)
    x_v = fuse_high_actv(1, to_high_actv(1, unfuse_low_actv(1, self.readin_v(x_v), self.L), self.L), self.L) # (L*batch, nodes, 96, 3)
    x_d = fuse_high_actv(2, to_high_actv(2, unfuse_low_actv(2, self.readin_d(x_d), self.L), self.L), self.L) # (L*batch, nodes, 64, 3, 3)
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, 8)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, 8, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, 8, 3, 3)
    x_a_j = edges_read_dst(graph, x_a) # (edges, 128)
    x_v_j = edges_read_dst(graph, x_v) # (edges, 96, 3)
    x_d_j = edges_read_dst(graph, x_d) # (edges, 64, 3, 3)
    ψ_a_ij, ψ_v_ij, ψ_d_ij = Ant16.apply(
      x_a_j, x_v_j, x_d_j,
      self.P_000, φ_a_ij, self.P_110, φ_v_ij, self.P_220, φ_d_ij,
      self.P_011, φ_a_ij, self.P_101, φ_v_ij, self.P_121, φ_v_ij, self.P_211, φ_d_ij, self.P_111, φ_v_ij,
      self.P_022, φ_a_ij, self.P_202, φ_d_ij, self.P_112, φ_v_ij, self.P_222, φ_d_ij, self.P_212, φ_d_ij)
    # do some edge-level MLPs cause why not
    ψ_a_ij = ψ_a_ij + self.mlp_a(ψ_a_ij) # edge-level MLP
    ψ_v_ij = ψ_v_ij + self.mlp_v(ψ_v_ij) # edge-level MLP
    # graph reduce
    B_a_i = edges_reduce_src(graph, ψ_a_ij) # (L*batch, nodes, 128)
    B_v_i = edges_reduce_src(graph, ψ_v_ij) # (L*batch, nodes, 96, 3)
    B_d_i = edges_reduce_src(graph, ψ_d_ij) # (L*batch, nodes, 64, 3, 3)
    # readout
    B_a_i = fuse_low_actv(0, to_low_actv(0, unfuse_high_actv(0, B_a_i, self.L), self.L), self.L) # (batch, nodes, L*128)
    B_v_i = fuse_low_actv(1, to_low_actv(1, unfuse_high_actv(1, B_v_i, self.L), self.L), self.L) # (batch, nodes, L*96, 3)
    B_d_i = fuse_low_actv(2, to_low_actv(2, unfuse_high_actv(2, B_d_i, self.L), self.L), self.L) # (batch, nodes, L*64, 3, 3)
    return self.readout_a(B_a_i), self.readout_v(B_v_i), self.readout_d(B_d_i)



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
  def __init__(self, config, L_0:int, L_1:int):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.lin_readin_a = TensLinear(0, dim_a, dim_a)
    self.lin_readin_v = TensLinear(1, dim_v, dim_v)
    self.lin_readin_d = TensLinear(2, dim_d, dim_d)
    self.disptens_0 = DisplacementTensors(config["r_cut"], 64, dim_a, dim_v, dim_d, L_0)
    self.disptens_1 = DisplacementTensors(config["r_cut"], 64, dim_a, dim_v, dim_d, L_1)
    self.prods = SelfTensProds(dim_a, dim_v, dim_d)
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


class LongResEnc(nn.Module):
  """ Residue position encoder for many positions. """
  def __init__(self, L, dim_v):
    super().__init__()
    self.L = L
    self.base_enc = ResiduesEncodeV2(dim_v)
    self.readout = TensLinear(1, L*dim_v, dim_v)
  def forward(self, pos, metadata):
    """ pos: (L, batch, atoms, 3) -- must be in high position """
    pos = fuse_high_pos(pos, self.L)
    centers, x_v = self.base_enc(pos, metadata) # (L*batch, aminos, 3), (L*batch, aminos, dim_v, 3)
    x_v = fuse_low_actv(1, to_low_actv(1, unfuse_high_actv(1, x_v, self.L), self.L), self.L)
    centers = unfuse_high_pos(centers, self.L)
    return centers, self.readout(x_v)


class AminoSubBlock(nn.Module):
  def __init__(self, config, L_0:int, L_1:int):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules:
    self.encode_0 = LongResEnc(L_0, dim_v)
    self.encode_1 = LongResEnc(L_1, dim_v)
    self.ndisp_embed = NeighbourDispEmbed(L_0, L_1, dim_v)
    self.pos_embed = PosEmbed(L_0, L_1, dim_v)
    self.readin_a = LinAtomToAmino(0, dim_a, dim_a)
    self.readin_v = LinAtomToAmino(1, dim_v, dim_v)
    self.readin_d = LinAtomToAmino(2, dim_d, dim_d)
    self.prods = SelfTensProds(dim_a, dim_v, dim_d)
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
  def __init__(self, config:Config, L_0:int, L_1:int, pos_1_mutable=False):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.pos_1_mutable:bool = pos_1_mutable
    self.r0:float = config["r_cut"] # cutoff radius for atom interactions
    self.L_0 = L_0
    self.L_1 = L_1
    # submodules:
    self.embed_t = TimeEmbedding(config["t_embed_hdim"], dim_a)
    self.pos_embed = PosEmbed(L_0, L_1, dim_v)
    self.res_embed = ResidueAtomEmbed(dim_a)
    self.sub_block = SubBlock(config, L_0, L_1)
    self.messages_0 = Messages(config["r_cut"], dim_a, dim_v, dim_d, L_0)
    self.messages_1 = Messages(config["r_cut"], dim_a, dim_v, dim_d, L_1)
    self.amino_sub_block = AminoSubBlock(config, L_0, L_1)
    if pos_1_mutable:
      self.lin_push_pos_1 = TensLinear(1, dim_v, L_1)
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
      pos_1 = pos_1 + to_high_pos(self.lin_push_pos_1(x_v), self.L_1)
      pos_1_tup = pos_1, None, None # invalidate previous graph and r_ij
    return t, contexttup, (x_a, x_v, x_d), ytup, pos_0_tup, pos_1_tup


class Denoiser(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.L_0 = config["framelen"]
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, self.L_0, 1, pos_1_mutable=True)
      for i in range(config["depth"])
    ])
  def forward(self, t, pos_0, pos_1, box, metadata):
    device = t.device
    must_be[self.L_0],  batch,          atoms,  must_be[3] = pos_0.shape
    must_be[1], must_be[batch], must_be[atoms], must_be[3] = pos_1.shape
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


def frameslices(framelen:int, x:torch.Tensor):
  """ x: (L, batch, ...)
      ans: (framelen, (L - framelen - 1)*batch, ...), ((L - framelen - 1)*batch, ...) """
  L, batch, *rest = x.shape
  dev = x.device
  indices_0 = torch.arange(L - framelen, device=dev)[None, :] + torch.arange(framelen, device=dev)[:, None]
  indices_1 = torch.arange(L - framelen, device=dev) + framelen
  x0 = x[indices_0] # (framelen, L - framelen, batch, ...)
  x1 = x[indices_1] # (L - framelen, batch, ...)
  return x0.reshape(framelen, (L - framelen)*batch, *rest), x1.reshape((L - framelen)*batch, *rest)


class ErrorTrackingScheduler:
  def __init__(self, lr:float, start_fac:float, warmup_steps:int, gamma:float):
    """ lr: float -- the base learning rate of this scheduler
        start_fac: float -- in [0, 1], initial learning rate is start_fac*lr
        warmup_steps: int -- number of steps for linear warmup from lr to start_fac
        gamma: float -- relative decay per step, decay across n steps is roughly (1 - gamma)**n
        * Usage is to construct scheduler before optimizer and initialize optimizer lr to scheduler.lr_prev """
    self.lr = lr
    self.start_fac = start_fac
    self.warmup_steps = warmup_steps
    self.gamma = gamma
    self.lr_prev = lr*start_fac
  def _curr_lr(self, step_count):
    t = step_count - self.warmup_steps
    return self.lr*min(
      1. + (1. - self.start_fac)*t/self.warmup_steps,
      np.exp(self.gamma*t)
    )
  def step(self, step_count, optim):
    lr_new = self._curr_lr(step_count)
    if abs(lr_new - self.lr_prev)/(lr_new + self.lr_prev) > 0.01:
      for group in optim.param_groups: # update the optimizer
        group["lr"] *= lr_new/self.lr_prev # relative update
      self.lr_prev = lr_new



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
    self.scheduler = ErrorTrackingScheduler(self.config["lr"], 0.1, 200, (1. - self.config["lr_fac"])/1024)
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim = torch.optim.AdamW(self.dn.parameters(),
      self.scheduler.lr_prev, betas, weight_decay=self.config["weight_decay"])
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
    self.scheduler.step(self.step_count, self.optim)
    return loss
  def sigma_t(self, t):
    return nodecay_cosine_schedule(t, self.sigma_max)
  def _diffuser_step(self, x, metadata):
    """ x: (L, batch, poly_len, 3) """
    L, batch, atoms, must_be[3] = x.shape
    framelen = self.config["framelen"]
    x_0, x_1 = frameslices(framelen, x) # (framelen, new_batch, atoms, 3), (new_batch, atoms, 3)
    t = torch.rand((L - framelen)*batch, device=x.device)
    epsilon = torch.randn_like(x_1)
    sigma = self.sigma_t(t)[:, None, None]
    x_1_noised = x_1 + sigma*epsilon
    x_1_pred = self.dn(t, x_0, x_1_noised[None], self.box, metadata).squeeze(0)
    loss = (((x_1_pred - x_1)/(0.4 + sigma))**2).mean()
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return loss.item()
  def generate(self, x_0, metadata, steps=12):
    framelen = self.config["framelen"]
    must_be[framelen], *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(framelen, batch, atoms, 3)
    ans = x_0[-1] + self.sigma_t(torch.zeros(1, device=x_0.device))[:, None, None]*self.randgen.randn(1, x_0.shape[1:-1])
    t_list = np.linspace(0., 1., steps + 1)
    for i in range(steps):
      t = torch.tensor([t_list[i]], device=x_0.device, dtype=torch.float32)
      tdec= torch.tensor([t_list[i + 1]], device=x_0.device, dtype=torch.float32)
      sigma_t = self.sigma_t(t)[:, None, None]
      sigma_tdec = self.sigma_t(tdec)[:, None, None]
      dsigma = torch.sqrt(sigma_t**2 - sigma_tdec**2)
      pred_noise = ans - self.dn(t, x_0, ans[None], self.box, metadata).squeeze(0)
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
