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
from managan.codegen_tensor_products import Ant16
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import (get_bond_graph, ResiduesEncode, ResidueEmbed, ResidueAtomEmbed,
      LinAminoToAtom, LinAtomToAmino, PosMix)
from managan.predictor import ModelState


# THIS FILE IS MOSTLY A CLONE OF diffuser_atoms.py (BACK QUITE A FEW ITERATIONS)
# IS JUST TO TEST WHAT HAPPENS WHEN WE SWAP OUT TENSOR PRODUCTS ONLY


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
  def __init__(self, dim_v):
    super().__init__()
    self.lin_v = TensLinear(1, 1, dim_v)
  def forward(self, pos_0, pos_1):
    """ pos_0, pos_1: (batch, nodes, 3) """
    pos_0, pos_1 = pos_0[:, :, None], pos_1[:, :, None]
    dpos_v = 0.1*(pos_1 - pos_0)
    return self.lin_v(dpos_v)


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


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij

def bond_r_ij_setup(graph:Graph, pos:torch.Tensor):
  pos_src, pos_dst = edges_read(graph, pos)
  return pos_dst - pos_src # since coordinates are unwrapped, we don't need boxwrap for bonded atoms

class Messages(nn.Module):
  def __init__(self, r0:float, dim_a:int, dim_v:int, dim_d:int, rank:int):
    super().__init__()
    print("Warning: rank param of Messages is dummy variable now, ignoring.")
    self.r0 = r0
    # parameters:
    add_tens_prod_parameters(self, dim_a, dim_v, dim_d, 8)
  def forward(self, graph, r_ij, x_a, x_v, x_d):
    rad_enc_ij = radial_encode_edge(r_ij, 8, self.r0)
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, 8)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, 8, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, 8, 3, 3)
    x_a_j = edges_read_dst(graph, x_a) # (edges, dim_a)
    x_v_j = edges_read_dst(graph, x_v) # (edges, dim_v, 3)
    x_d_j = edges_read_dst(graph, x_d) # (edges, dim_d, 3, 3)
    ψ_a_ij, ψ_v_ij, ψ_d_ij = Ant16.apply(
      x_a_j, x_v_j, x_d_j,
      self.P_000, φ_a_ij, self.P_110, φ_v_ij, self.P_220, φ_d_ij,
      self.P_011, φ_a_ij, self.P_101, φ_v_ij, self.P_121, φ_v_ij, self.P_211, φ_d_ij, self.P_111, φ_v_ij,
      self.P_022, φ_a_ij, self.P_202, φ_d_ij, self.P_112, φ_v_ij, self.P_222, φ_d_ij, self.P_212, φ_d_ij)
    B_a_i = edges_reduce_src(graph, ψ_a_ij) # (batch, nodes, dim_a)
    B_v_i = edges_reduce_src(graph, ψ_v_ij) # (batch, nodes, dim_v, 3)
    B_d_i = edges_reduce_src(graph, ψ_d_ij) # (batch, nodes, dim_d, 3, 3)
    return 0.1*B_a_i, 0.1*B_v_i, 0.1*B_d_i


class LocalMLP(nn.Module):
  """ Tensor MLP that acts locally. Has a built-in residual connection. """
  def __init__(self, inds, dim, t_embed_hdim):
    super().__init__()
    self.lin1 = TensLinear(inds, dim, dim)
    self.actv1 = nn.LeakyReLU(0.1) if inds == 0 else TensSigmoid(inds)
    self.lin2 = TimeLinearModulation(t_embed_hdim, inds, dim)
    self.actv2 = nn.LeakyReLU(0.1) if inds == 0 else TensSigmoid(inds)
    self.lin3 = TensLinear(inds, dim, dim)
  def forward(self, t, x):
    z = self.actv1(self.lin1(x))
    z = self.actv2(self.lin2(z, t))
    return self.lin3(x + z)


class AVDLocalMLPs(nn.Module):
  """ Update state locally. """
  def __init__(self, config, dim_a:int, dim_v:int, dim_d:int):
    super().__init__()
    # submodules:
    self.linear_mix = AVDFullLinearMix(dim_a, dim_v, dim_d)
    self.tens_prods = SelfTensProds(dim_a, dim_v, dim_d)
    self.gn_a = TensGroupNormBroken(0, dim_a, config["groups_a"])
    self.gn_v = TensGroupNormBroken(1, dim_v, config["groups_v"])
    self.gn_d = TensGroupNormBroken(2, dim_d, config["groups_d"])
    self.mlp_a = LocalMLP(0, dim_a, config["t_embed_hdim"])
    self.mlp_v = LocalMLP(1, dim_v, config["t_embed_hdim"])
    self.mlp_d = LocalMLP(2, dim_d, config["t_embed_hdim"])
  def forward(self, t, x_a, x_v, x_d):
    # linear stuff like swapping indices and contracting with LC-symbol
    Δx_a, Δx_v, Δx_d = self.linear_mix(x_a, x_v, x_d)
    x_a, x_v, x_d = x_a + 0.1*Δx_a, x_v + 0.1*Δx_v, x_d + 0.1*Δx_d
    # tensor products
    Δx_a, Δx_v, Δx_d = self.tens_prods(0.2*x_a, 0.2*x_v, 0.2*x_d) # scale down to prevent values from blowing up
    x_a, x_v, x_d = x_a + Δx_a, x_v + Δx_v, x_d + Δx_d
    # multilayer perceptrons
    x_a = self.mlp_a(t, x_a)
    x_v = self.mlp_v(t, x_v)
    x_d = self.mlp_d(t, x_d)
    return self.gn_a(x_a), self.gn_v(x_v), self.gn_d(x_d)

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
  def forward(self, x_a, x_v, x_d):
    """ x_a, x_v, x_d: (batch, nodes, dim, (3,)^inds """
    x_a = self.convs_a(x_a)
    x_v = self.convs_v(x_v)
    x_d = self.convs_d(x_d)
    return x_a, x_v, x_d


class Block(nn.Module):
  """ Processing block for nets. """
  def __init__(self, config:Config):
    super().__init__()
    dim_a_atm, dim_v_atm, dim_d_atm = config["dim_a_atm"], config["dim_v_atm"], config["dim_d_atm"]
    dim_a_amn, dim_v_amn, dim_d_amn = config["dim_a_amn"], config["dim_v_amn"], config["dim_d_amn"]
    self.r0_atm:float = config["r_cut_atm"] # cutoff radius for atom interactions
    self.r0_amn:float = config["r_cut_amn"] # cutoff radius for amino interactions
    self.r0_bond:float = config["r_cut_bond"] # cutoff radius for bonded atom interactions. should probably match r0_atm.
    # submodules:
    self.embed_t_atm = TimeEmbedding(config["t_embed_hdim"], dim_a_atm)
    self.embed_t_amn = TimeEmbedding(config["t_embed_hdim"], dim_a_amn)
    self.arc_embed = ArcEmbed(dim_a_amn)
    self.pos_embed_atm = PosEmbed(dim_v_atm)
    self.pos_embed_amn = PosEmbed(dim_v_amn)
    self.ndisp_embed1 = NeighbourDispEmbed(dim_v_amn)
    self.ndisp_embed2 = NeighbourDispEmbed(dim_v_amn)
    self.res_embed_atm = ResidueAtomEmbed(dim_a_atm)
    self.res_embed_amn = ResidueEmbed(dim_a_amn)
    self.local1_atm = AVDLocalMLPs(config, dim_a_atm, dim_v_atm, dim_d_atm)
    self.local1_amn = AVDLocalMLPs(config, dim_a_amn, dim_v_amn, dim_d_amn)
    self.push1_pos_0_atm = TensLinear(1, dim_v_atm, 1)
    self.push1_pos_1_atm = TensLinear(1, dim_v_atm, 1)
    self.push1_pos_0_amn = TensLinear(1, dim_v_amn, 1)
    self.push1_pos_1_amn = TensLinear(1, dim_v_amn, 1)
    self.messages_bond_0 = Messages(self.r0_bond, dim_a_atm, dim_v_atm, dim_d_atm, config["rank"])
    self.messages_bond_1 = Messages(self.r0_bond, dim_a_atm, dim_v_atm, dim_d_atm, config["rank"])
    self.messages_0_atm = Messages(self.r0_atm, dim_a_atm, dim_v_atm, dim_d_atm, config["rank"])
    self.messages_1_atm = Messages(self.r0_atm, dim_a_atm, dim_v_atm, dim_d_atm, config["rank"])
    self.messages_0_amn = Messages(self.r0_amn, dim_a_amn, dim_v_amn, dim_d_amn, config["rank"])
    self.messages_1_amn = Messages(self.r0_amn, dim_a_amn, dim_v_amn, dim_d_amn, config["rank"])
    self.conv_mlps = AminoChainConv(dim_a_amn, dim_v_amn, dim_d_amn)
    self.pos_mix_0 = PosMix(dim_v_atm, dim_v_amn)
    self.pos_mix_1 = PosMix(dim_v_atm, dim_v_amn)
    self.lin_a_atm_to_amn = LinAtomToAmino(0, dim_a_atm, dim_a_amn)
    self.lin_v_atm_to_amn = LinAtomToAmino(1, dim_v_atm, dim_v_amn)
    self.lin_d_atm_to_amn = LinAtomToAmino(2, dim_d_atm, dim_d_amn)
    self.lin_a_amn_to_atm = LinAminoToAtom(0, dim_a_amn, dim_a_atm)
    self.lin_v_amn_to_atm = LinAminoToAtom(1, dim_v_amn, dim_v_atm)
    self.lin_d_amn_to_atm = LinAminoToAtom(2, dim_d_amn, dim_d_atm)
    self.local2_atm = AVDLocalMLPs(config, dim_a_atm, dim_v_atm, dim_d_atm)
    self.local2_amn = AVDLocalMLPs(config, dim_a_amn, dim_v_amn, dim_d_amn)
    self.push2_pos_0_atm = TensLinear(1, dim_v_atm, 1)
    self.push2_pos_1_atm = TensLinear(1, dim_v_atm, 1)
    self.push2_pos_0_amn = TensLinear(1, dim_v_amn, 1)
    self.push2_pos_1_amn = TensLinear(1, dim_v_amn, 1)
  def self_init(self):
    with torch.no_grad():
      self.push1_pos_0_atm.W.zero_()
      self.push1_pos_1_atm.W.zero_()
      self.push1_pos_0_amn.W.zero_()
      self.push1_pos_1_amn.W.zero_()
      self.push2_pos_0_atm.W.zero_()
      self.push2_pos_1_atm.W.zero_()
      self.push2_pos_0_amn.W.zero_()
      self.push2_pos_1_amn.W.zero_()
  def forward(self, tup):
    # unpack the tuple
    t, contexttup, datatup_atm, datatup_amn = tup
    pos_0_atm, pos_1_atm, x_a_atm, x_v_atm, x_d_atm = datatup_atm
    pos_0_amn, pos_1_amn, x_a_amn, x_v_amn, x_d_amn = datatup_amn
    bond_graph, box, metadata = contexttup
    # embed and initial MLP's
    x_v_amn = x_v_amn + self.ndisp_embed1(pos_0_amn, pos_1_amn)
    dx_a_atm, dx_v_atm, dx_d_atm = self.local1_atm(t,
      x_a_atm + self.embed_t_atm(t)[:, None] + self.res_embed_atm(metadata),
      x_v_atm + self.pos_embed_atm(pos_0_atm, pos_1_atm),
      x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    dx_a_amn, dx_v_amn, dx_d_amn = self.local1_amn(t,
      (x_a_amn + self.embed_t_amn(t)[:, None] + self.res_embed_amn(metadata)
        + self.arc_embed(pos_0_amn.shape[0], pos_0_amn.shape[1], pos_0_amn.device)),
      x_v_amn + self.pos_embed_amn(pos_0_amn, pos_1_amn) + self.ndisp_embed2(pos_0_amn, pos_1_amn),
      x_d_amn)
    x_a_amn, x_v_amn, x_d_amn = x_a_amn + dx_a_amn, x_v_amn + dx_v_amn, x_d_amn + dx_d_amn # AMN UPDATE
    # push positions before graph ops
    pos_0_atm = pos_0_atm + self.push1_pos_0_atm(x_v_atm).squeeze(2)
    pos_1_atm = pos_1_atm + self.push1_pos_1_atm(x_v_atm).squeeze(2)
    pos_0_amn = pos_0_amn + self.push1_pos_0_amn(x_v_amn).squeeze(2)
    pos_1_amn = pos_1_amn + self.push1_pos_1_amn(x_v_amn).squeeze(2)
    # messages with bond graph
    bond_r_ij_0 = bond_r_ij_setup(bond_graph, pos_0_atm)
    dx_a_atm, dx_v_atm, dx_d_atm = self.messages_bond_0(bond_graph, bond_r_ij_0, x_a_atm, x_v_atm, x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    bond_r_ij_1 = bond_r_ij_setup(bond_graph, pos_1_atm)
    dx_a_atm, dx_v_atm, dx_d_atm = self.messages_bond_1(bond_graph, bond_r_ij_1, x_a_atm, x_v_atm, x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    # messages with proximity graphs
    graph_0_atm, r_ij_0_atm = graph_setup(self.r0_atm, box, pos_0_atm)
    dx_a_atm, dx_v_atm, dx_d_atm = self.messages_0_atm(graph_0_atm, r_ij_0_atm, x_a_atm, x_v_atm, x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    graph_1_atm, r_ij_1_atm = graph_setup(self.r0_atm, box, pos_1_atm)
    dx_a_atm, dx_v_atm, dx_d_atm = self.messages_1_atm(graph_1_atm, r_ij_1_atm, x_a_atm, x_v_atm, x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    graph_0_amn, r_ij_0_amn = graph_setup(self.r0_amn, box, pos_0_amn)
    dx_a_amn, dx_v_amn, dx_d_amn = self.messages_0_amn(graph_0_amn, r_ij_0_amn, x_a_amn, x_v_amn, x_d_amn)
    x_a_amn, x_v_amn, x_d_amn = x_a_amn + dx_a_amn, x_v_amn + dx_v_amn, x_d_amn + dx_d_amn # AMN UPDATE
    graph_1_amn, r_ij_1_amn = graph_setup(self.r0_amn, box, pos_1_amn)
    dx_a_amn, dx_v_amn, dx_d_amn = self.messages_1_amn(graph_1_amn, r_ij_1_amn, x_a_amn, x_v_amn, x_d_amn)
    x_a_amn, x_v_amn, x_d_amn = x_a_amn + dx_a_amn, x_v_amn + dx_v_amn, x_d_amn + dx_d_amn # AMN UPDATE
    # convolutions
    dx_a_amn, dx_v_amn, dx_d_amn = self.conv_mlps(x_a_amn, x_v_amn, x_d_amn)
    x_a_amn, x_v_amn, x_d_amn = x_a_amn + dx_a_amn, x_v_amn + dx_v_amn, x_d_amn + dx_d_amn # AMN UPDATE
    # pass data between amino and atom levels
    dx_v_atm_0, dx_v_amn_0 = self.pos_mix_0(pos_0_atm, pos_0_amn, metadata)
    dx_v_atm_1, dx_v_amn_1 = self.pos_mix_1(pos_1_atm, pos_1_amn, metadata)
    x_v_atm = x_v_atm + dx_v_atm_0 + dx_v_atm_1
    x_v_amn = x_v_amn + dx_v_amn_0 + dx_v_amn_1
    x_a_amn = x_a_amn + 0.1*self.lin_a_atm_to_amn(x_a_atm, metadata)
    x_v_amn = x_v_amn + 0.1*self.lin_v_atm_to_amn(x_v_atm, metadata)
    x_d_amn = x_d_amn + 0.1*self.lin_d_atm_to_amn(x_d_atm, metadata)
    x_a_atm = x_a_atm + 0.3*self.lin_a_amn_to_atm(x_a_amn, metadata)
    x_v_atm = x_v_atm + 0.3*self.lin_v_amn_to_atm(x_v_amn, metadata)
    x_d_atm = x_d_atm + 0.3*self.lin_d_amn_to_atm(x_d_amn, metadata)
    # final MLP's
    dx_a_atm, dx_v_atm, dx_d_atm = self.local2_atm(t, x_a_atm, x_v_atm, x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    dx_a_amn, dx_v_amn, dx_d_amn = self.local2_amn(t, x_a_amn, x_v_amn, x_d_amn)
    x_a_amn, x_v_amn, x_d_amn = x_a_amn + dx_a_amn, x_v_amn + dx_v_amn, x_d_amn + dx_d_amn # AMN UPDATE
    # final push positions
    pos_0_atm = pos_0_atm + self.push2_pos_0_atm(x_v_atm).squeeze(2)
    pos_1_atm = pos_1_atm + self.push2_pos_1_atm(x_v_atm).squeeze(2)
    pos_0_amn = pos_0_amn + self.push2_pos_0_amn(x_v_amn).squeeze(2)
    pos_1_amn = pos_1_amn + self.push2_pos_1_amn(x_v_amn).squeeze(2)
    # repack the tuple and RETURN
    datatup_atm = pos_0_atm, pos_1_atm, x_a_atm, x_v_atm, x_d_atm
    datatup_amn = pos_0_amn, pos_1_amn, x_a_amn, x_v_amn, x_d_amn
    return t, contexttup, datatup_atm, datatup_amn


class Denoiser(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a_atm, self.dim_v_atm, self.dim_d_atm = config["dim_a_atm"], config["dim_v_atm"], config["dim_d_atm"]
    self.dim_a_amn, self.dim_v_amn, self.dim_d_amn = config["dim_a_amn"], config["dim_v_amn"], config["dim_d_amn"]
    self.encode_0 = ResiduesEncode(self.dim_v_amn, nlin=2)
    self.encode_1 = ResiduesEncode(self.dim_v_amn, nlin=2)
    self.blocks = nn.Sequential(*[
      Block(config)
      for i in range(config["depth"])
    ])
  def forward(self, t, pos_0_atm, pos_1_atm, box, metadata):
    device = t.device
    batch,          atoms,          must_be[3] = pos_0_atm.shape
    must_be[batch], must_be[atoms], must_be[3] = pos_1_atm.shape
    nodes = len(metadata.seq)
    pos_0_amn, x_v_0_amn = self.encode_0(pos_0_atm, metadata)
    pos_1_amn, x_v_1_amn = self.encode_1(pos_1_atm, metadata)
    x_a_amn = torch.zeros(batch, nodes, self.dim_a_amn, device=device)
    x_v_amn = x_v_0_amn + x_v_1_amn
    x_d_amn = torch.zeros(batch, nodes, self.dim_d_amn, 3, 3, device=device)
    x_a_atm = torch.zeros(batch, atoms, self.dim_a_atm, device=device)
    x_v_atm = torch.zeros(batch, atoms, self.dim_v_atm, 3, device=device)
    x_d_atm = torch.zeros(batch, atoms, self.dim_d_atm, 3, 3, device=device)
    # get the graph of bonds for this network
    bond_graph = get_bond_graph(batch, metadata, device)
    # run the main network
    contexttup = bond_graph, box, metadata
    datatup_atm = pos_0_atm, pos_1_atm, x_a_atm, x_v_atm, x_d_atm
    datatup_amn = pos_0_amn, pos_1_amn, x_a_amn, x_v_amn, x_d_amn
    tup = t, contexttup, datatup_atm, datatup_amn
    tup = self.blocks(tup)
    t, contexttup, datatup_atm, datatup_amn = tup
    pos_0_atm, pos_1_atm, x_a_atm, x_v_atm, x_d_atm = datatup_atm
    return pos_1_atm


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
