from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod, typsz
from managan.layers_common import *
from managan.config import Config, load
from managan.tensor_products import *
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import (get_bond_graph, ResiduesEncode, ResidueEmbed, ResidueAtomEmbed,
      LinAminoToAtom, LinAtomToAmino, PosMix)
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
    self.tp_220 = TensorProds(2, 2, 0, dim_d, 8, dim_a, rank)
    self.tp_222 = TensorProds(2, 2, 2, dim_d, 8, dim_d, rank)
  def forward(self, graph, r_ij, x_a, x_v, x_d):
    rad_enc_ij = radial_encode_edge(r_ij, 8, self.r0)
    r_ij = tens_sigmoid(1, r_ij*(17./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                                                        # (edges, 8)
    φ_v_ij = rad_enc_ij[..., None]       * r_ij[..., None, :]                                  # (edges, 8, 3)
    φ_d_ij = rad_enc_ij[..., None, None] * r_ij[..., None, None, :] * r_ij[..., None, :, None] # (edges, 8, 3, 3)
    x_a_j = edges_read_dst(graph, x_a) # (edges, dim_a)
    x_v_j = edges_read_dst(graph, x_v) # (edges, dim_v, 3)
    x_d_j = edges_read_dst(graph, x_d) # (edges, dim_d, 3, 3)
    ψ_a_ij = self.tp_000(x_a_j, φ_a_ij) + self.tp_110(x_v_j, φ_v_ij) + self.tp_220(x_d_j, φ_d_ij)
    ψ_v_ij = self.tp_011(x_a_j, φ_v_ij) + self.tp_101(x_v_j, φ_a_ij) + self.tp_211(x_d_j, φ_v_ij)
    ψ_d_ij = self.tp_112(x_v_j, φ_v_ij) + self.tp_202(x_d_j, φ_a_ij) + self.tp_222(x_d_j, φ_d_ij)
    B_a_i = edges_reduce_src(graph, ψ_a_ij) # (batch, nodes, dim_a)
    B_v_i = edges_reduce_src(graph, ψ_v_ij) # (batch, nodes, dim_v, 3)
    B_d_i = edges_reduce_src(graph, ψ_d_ij) # (batch, nodes, dim_d, 3, 3)
    return B_a_i, B_v_i, B_d_i


class LocalMLP(nn.Module):
  """ Tensor MLP that acts locally. Has a built-in residual connection. """
  def __init__(self, inds, dim):
    super().__init__()
    self.lin1 = TensLinear(inds, dim, dim)
    self.actv1 = nn.LeakyReLU(0.1) if inds == 0 else TensSigmoid(inds)
    self.lin2 = TensLinear(inds, dim, dim)
    self.actv2 = nn.LeakyReLU(0.1) if inds == 0 else TensSigmoid(inds)
    self.lin3 = TensLinear(inds, dim, dim)
  def forward(self, x):
    z = self.actv1(self.lin1(x))
    z = self.actv2(self.lin2(z))
    return self.lin3(x + z)


class AVDLocalMLPs(nn.Module):
  """ Update state locally. """
  def __init__(self, config, dim_a:int, dim_v:int, dim_d:int):
    super().__init__()
    # submodules:
    self.linear_mix = AVDFullLinearMix(dim_a, dim_v, dim_d)
    self.tens_prods = AVDFullTensorProds(dim_a, dim_v, dim_d, config["rank"])
    self.gn_a = TensGroupNormBroken(0, dim_a, config["groups_a"])
    self.gn_v = TensGroupNormBroken(1, dim_v, config["groups_v"])
    self.gn_d = TensGroupNormBroken(2, dim_d, config["groups_d"])
    self.mlp_a = LocalMLP(0, dim_a)
    self.mlp_v = LocalMLP(1, dim_v)
    self.mlp_d = LocalMLP(2, dim_d)
  def forward(self, x_a, x_v, x_d):
    # linear stuff like swapping indices and contracting with LC-symbol
    Δx_a, Δx_v, Δx_d = self.linear_mix(x_a, x_v, x_d)
    x_a, x_v, x_d = x_a + 0.1*Δx_a, x_v + 0.1*Δx_v, x_d + 0.1*Δx_d
    # tensor products
    Δx_a, Δx_v, Δx_d = self.tens_prods(0.2*x_a, 0.2*x_v, 0.2*x_d) # scale down to prevent values from blowing up
    x_a, x_v, x_d = x_a + Δx_a, x_v + Δx_v, x_d + Δx_d
    # multilayer perceptrons
    x_a = self.mlp_a(x_a)
    x_v = self.mlp_v(x_v)
    x_d = self.mlp_d(x_d)
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


class XToEnergy(nn.Module):
  """ Convert activations into a single energy. """
  def __init__(self, dim_a):
    super().__init__()
    self.E_out_lin_a = TensLinear(0, dim_a, dim_a)
    self.E_out_readout_a = TensLinear(0, dim_a, 1)
  def self_init(self):
    with torch.no_grad():
      self.E_out_lin_a.W.mul_(0.1)
  def forward(self, x_a):
    """ x_a: (batch, nodes, dim_a)
        ans: (batch, nodes) """
    return self.E_out_readout_a(torch.relu(self.E_out_lin_a(x_a))).squeeze(-1)


class Block(nn.Module):
  """ Processing block for nets. """
  def __init__(self, config:Config, randgen:TensorRandGen, is_disc:bool):
    super().__init__()
    self.randgen = randgen
    self.is_disc = is_disc
    dim_a_atm, dim_v_atm, dim_d_atm = config["dim_a_atm"], config["dim_v_atm"], config["dim_d_atm"]
    dim_a_amn, dim_v_amn, dim_d_amn = config["dim_a_amn"], config["dim_v_amn"], config["dim_d_amn"]
    self.r0_atm:float = config["r_cut_atm"] # cutoff radius for atom interactions
    self.r0_amn:float = config["r_cut_amn"] # cutoff radius for amino interactions
    self.r0_bond:float = config["r_cut_bond"] # cutoff radius for bonded atom interactions. should probably match r0_atm.
    # submodules:
    if not self.is_disc:
      self.rand_lin_a_atm = TensLinear(0, dim_a_atm, dim_a_atm)
      self.rand_lin_v_atm = TensLinear(1, dim_v_atm, dim_v_atm)
      self.rand_lin_d_atm = TensLinear(2, dim_d_atm, dim_d_atm)
      self.rand_lin_a_amn = TensLinear(0, dim_a_amn, dim_a_amn)
      self.rand_lin_v_amn = TensLinear(1, dim_v_amn, dim_v_amn)
      self.rand_lin_d_amn = TensLinear(2, dim_d_amn, dim_d_amn)
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
    if self.is_disc:
      self.x_to_E_atm = XToEnergy(dim_a_atm)
      self.x_to_E_amn = XToEnergy(dim_a_amn)
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
    contexttup, datatup_atm, datatup_amn = tup
    pos_0_atm, pos_1_atm, E_atm, x_a_atm, x_v_atm, x_d_atm = datatup_atm
    pos_0_amn, pos_1_amn, E_amn, x_a_amn, x_v_amn, x_d_amn = datatup_amn
    bond_graph, box, metadata = contexttup
    # add noise
    if not self.is_disc:
      x_a_atm = x_a_atm + self.rand_lin_a_atm(self.randgen.randn(0, x_a_atm.shape))
      x_v_atm = x_v_atm + self.rand_lin_v_atm(self.randgen.randn(1, x_v_atm.shape[:-1]))
      x_d_atm = x_d_atm + self.rand_lin_d_atm(self.randgen.randn(2, x_d_atm.shape[:-2]))
      x_a_amn = x_a_amn + self.rand_lin_a_amn(self.randgen.randn(0, x_a_amn.shape))
      x_v_amn = x_v_amn + self.rand_lin_v_amn(self.randgen.randn(1, x_v_amn.shape[:-1]))
      x_d_amn = x_d_amn + self.rand_lin_d_amn(self.randgen.randn(2, x_d_amn.shape[:-2]))
    # embed and initial MLP's
    x_v_amn = x_v_amn + self.ndisp_embed1(pos_0_amn, pos_1_amn)
    dx_a_atm, dx_v_atm, dx_d_atm = self.local1_atm(
      x_a_atm + self.res_embed_atm(metadata),
      x_v_atm + self.pos_embed_atm(pos_0_atm, pos_1_atm),
      x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    dx_a_amn, dx_v_amn, dx_d_amn = self.local1_amn(
      (x_a_amn + self.res_embed_amn(metadata)
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
    dx_a_atm, dx_v_atm, dx_d_atm = self.local2_atm(x_a_atm, x_v_atm, x_d_atm)
    x_a_atm, x_v_atm, x_d_atm = x_a_atm + dx_a_atm, x_v_atm + dx_v_atm, x_d_atm + dx_d_atm # ATM UPDATE
    dx_a_amn, dx_v_amn, dx_d_amn = self.local2_amn(x_a_amn, x_v_amn, x_d_amn)
    x_a_amn, x_v_amn, x_d_amn = x_a_amn + dx_a_amn, x_v_amn + dx_v_amn, x_d_amn + dx_d_amn # AMN UPDATE
    # final push positions
    pos_0_atm = pos_0_atm + self.push2_pos_0_atm(x_v_atm).squeeze(2)
    pos_1_atm = pos_1_atm + self.push2_pos_1_atm(x_v_atm).squeeze(2)
    pos_0_amn = pos_0_amn + self.push2_pos_0_amn(x_v_amn).squeeze(2)
    pos_1_amn = pos_1_amn + self.push2_pos_1_amn(x_v_amn).squeeze(2)
    # update energies
    if self.is_disc:
      E_atm = E_atm + self.x_to_E_atm(x_a_atm)
      E_amn = E_amn + self.x_to_E_amn(x_a_amn)
    # repack the tuple and RETURN
    datatup_atm = pos_0_atm, pos_1_atm, E_atm, x_a_atm, x_v_atm, x_d_atm
    datatup_amn = pos_0_amn, pos_1_amn, E_amn, x_a_amn, x_v_amn, x_d_amn
    return contexttup, datatup_atm, datatup_amn


class DiscHead(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a_atm, dim_v_atm, dim_d_atm = config["dim_a_atm"], config["dim_v_atm"], config["dim_d_atm"]
    dim_a_amn, dim_v_amn, dim_d_amn = config["dim_a_amn"], config["dim_v_amn"], config["dim_d_amn"]
    # submodules
    self.pos_embed_atm = PosEmbed(dim_v_atm)
    self.pos_embed_amn = PosEmbed(dim_v_amn)
    self.lin_a_atm = TensLinear(0, dim_v_atm, dim_a_atm)
    self.lin_a_amn = TensLinear(0, dim_v_amn, dim_a_amn)
    self.x_to_E_atm = XToEnergy(dim_a_atm)
    self.x_to_E_amn = XToEnergy(dim_a_amn)
  def forward(self, tup):
    # unpack the tuple
    contexttup, datatup_atm, datatup_amn = tup
    pos_0_atm, pos_1_atm, E_atm, x_a_atm, x_v_atm, x_d_atm = datatup_atm
    pos_0_amn, pos_1_amn, E_amn, x_a_amn, x_v_amn, x_d_amn = datatup_amn
    # update x_v activations based on positions
    x_v_atm = x_v_atm + self.pos_embed_atm(pos_0_atm, pos_1_atm)
    x_v_amn = x_v_amn + self.pos_embed_amn(pos_0_amn, pos_1_amn)
    # update x_a activations
    x_a_atm = x_a_atm + self.lin_a_atm(torch.linalg.vector_norm(x_v_atm, dim=-1))
    x_a_amn = x_a_amn + self.lin_a_amn(torch.linalg.vector_norm(x_v_amn, dim=-1))
    # final update of energies
    E_atm = E_atm + self.x_to_E_atm(x_a_atm)
    E_amn = E_amn + self.x_to_E_amn(x_a_amn)
    return E_atm.mean(-1) + E_amn.mean(-1) # average over nodes and atoms
'''
class DiscHead(nn.Module):
  def forward(self, tup):
    # unpack the tuple
    contexttup, datatup_atm, datatup_amn = tup
    pos_0_atm, pos_1_atm, E_atm, x_a_atm, x_v_atm, x_d_atm = datatup_atm
    pos_0_amn, pos_1_amn, E_amn, x_a_amn, x_v_amn, x_d_amn = datatup_amn
    return E_atm.mean(-1) + E_amn.mean(-1) # average over nodes and atoms
'''


class Discriminator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a_atm, self.dim_v_atm, self.dim_d_atm = config["dim_a_atm"], config["dim_v_atm"], config["dim_d_atm"]
    self.dim_a_amn, self.dim_v_amn, self.dim_d_amn = config["dim_a_amn"], config["dim_v_amn"], config["dim_d_amn"]
    self.encode_0 = ResiduesEncode(self.dim_v_amn, nlin=2)
    self.encode_1 = ResiduesEncode(self.dim_v_amn, nlin=2)
    self.blocks = nn.Sequential(*[
      Block(config, randgen, True)
      for i in range(config["depth"])
    ], DiscHead(config))
  def forward(self, pos_0_atm, pos_1_atm, box, metadata):
    device = pos_0_atm.device
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
    datatup_atm = pos_0_atm, pos_1_atm, 0., x_a_atm, x_v_atm, x_d_atm
    datatup_amn = pos_0_amn, pos_1_amn, 0., x_a_amn, x_v_amn, x_d_amn
    tup = contexttup, datatup_atm, datatup_amn
    return self.blocks(tup)


class Generator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a_atm, self.dim_v_atm, self.dim_d_atm = config["dim_a_atm"], config["dim_v_atm"], config["dim_d_atm"]
    self.dim_a_amn, self.dim_v_amn, self.dim_d_amn = config["dim_a_amn"], config["dim_v_amn"], config["dim_d_amn"]
    self.encode_0 = ResiduesEncode(self.dim_v_amn, nlin=2)
    self.encode_1 = ResiduesEncode(self.dim_v_amn, nlin=2)
    self.blocks = nn.Sequential(*[
      Block(config, randgen, False)
      for i in range(config["depth"])
    ])
  def forward(self, pos_0_atm, box, metadata):
    device = pos_0_atm.device
    batch,          atoms,          must_be[3] = pos_0_atm.shape
    nodes = len(metadata.seq)
    pos_1_atm = pos_0_atm
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
    datatup_atm = pos_0_atm, pos_1_atm, 0., x_a_atm, x_v_atm, x_d_atm
    datatup_amn = pos_0_amn, pos_1_amn, 0., x_a_amn, x_v_amn, x_d_amn
    tup = contexttup, datatup_atm, datatup_amn
    tup = self.blocks(tup)
    contexttup, datatup_atm, datatup_amn = tup
    pos_0_atm, pos_1_atm, E_atm, x_a_atm, x_v_atm, x_d_atm = datatup_atm
    return norm_grad(pos_1_atm)


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
  x_1 = mix_factors_1*x_g + (1 - mix_factors_1)*x_r
  y_1 = disc(x_0, x_1, box, metadata)
  return (endpoint_penalty(x_r, x_g, y_r, y_g)
        + endpoint_penalty(x_r, x_1, y_r, y_1)
        + endpoint_penalty(x_g, x_1, y_g, y_1))


class WGAN3D:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.config = config
    self.discs = []
    for _ in range(config["ndiscs"]):
      self.add_new_disc()
    self.gen = Generator(config, self.randgen).to(config.device)
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
        sigma = model.sigma_t(t)[:, None, None]
        x_1_g_noised = x_1_g + sigma*epsilon
        x_1_g_pred = model.dn(t, x_0, x_1_g_noised, self.box, metadata)
        loss = loss + self.config["distill_lambdas"][nsteps]*(((x_1_g_pred - x_1_g.detach())/(0.4 + sigma))**2).mean()
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
