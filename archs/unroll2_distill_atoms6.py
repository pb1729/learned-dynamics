from typing_extensions import Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod, turn_on_actv_size_printing
from managan.layers_common import *
from managan.config import Config, load
from managan.tensor_products import TensLinear, TensConv1d, tens_sigmoid, TensSigmoid, TensGroupNorm, TensorRandGen
from managan.codegen_tensor_products import Bee
from managan.jacobi_radenc import radial_encode_8
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResiduesEncodeV2, ResidueAtomEmbed, LinAminoToAtom, LinAtomToAmino
from managan.learn_common import ErrorTrackingScheduler, get_lr_fn_warmup_and_decay, get_endpt_pen, norm_grad
from managan.predictor import ModelState


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
      nn.Linear(dim_a, dim_a),
      nn.LeakyReLU(0.1),
      nn.Linear(dim_a, dim_a),
      nn.LeakyReLU(0.1),
      nn.Linear(dim_a, dim_a),
    )
  def forward(self, x):
    return self.lin_direct(x) + self.layers(x)

class VectorMLP(nn.Module):
  def __init__(self, dim_v):
    super().__init__()
    self.lin_direct = TensLinear(1, dim_v, dim_v)
    self.layers = nn.Sequential(
      TensLinear(1, dim_v, dim_v),
      TensSigmoid(1),
      TensLinear(1, dim_v, dim_v),
      TensSigmoid(1),
      TensLinear(1, dim_v, dim_v),
    )
  def forward(self, x):
    return self.lin_direct(x) + self.layers(x)


class LocalMLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules:
    self.mlp_a = torch.compile(MLP(dim_a))
    self.mlp_v = VectorMLP(dim_v)
    self.lin_d_trans = TensLinear(2, dim_d, dim_d)
  def forward(self, x_a, x_v, x_d):
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
    contexttup, xtup, pos_0_tup, pos_1_tup = tup
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
    return self.mlps_out(x_a, x_v, x_d)


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
    contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
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
    Δy_a, Δy_v, Δy_d = self.mlps_out(*self.prods(y_a, y_v, y_d))
    y_a = y_a + Δy_a
    y_v = y_v + Δy_v
    y_d = y_d + Δy_d
    # readout
    Δxtup = 0.2*self.readout_a(y_a, metadata), 0.2*self.readout_v(y_v, metadata), 0.2*self.readout_d(y_d, metadata)
    return Δxtup, (y_a, y_v, y_d)


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
  """ Processing block for nets. """
  K_DISC = 0
  K_GEN = 1
  def __init__(self, config:Config, kind:int, randgen:Union[None, TensorRandGen]):
    super().__init__()
    dim_a, dim_v, dim_d, chan = config["dim_a"], config["dim_v"], config["dim_d"], config["chan"]
    self.kind:int = kind
    self.pos_1_mutable:bool = (kind == self.K_GEN)
    self.r0:float = config["r_cut"] # cutoff radius for atom interactions
    if "neighbours_max" in config:
      self.neighbours_max = config["neighbours_max"]
    else:
      self.neighbours_max = 64
    self.randgen = randgen
    # submodules:
    self.pos_embed = PosEmbed(dim_v)
    self.res_embed = ResidueAtomEmbed(dim_a)
    self.sub_block = SubBlock(config)
    self.messages_0 = Messages(config["r_cut"], dim_a, dim_v, dim_d, chan)
    self.messages_1 = Messages(config["r_cut"], dim_a, dim_v, dim_d, chan)
    self.amino_sub_block = AminoSubBlock(config)
    if self.pos_1_mutable:
      self.lin_push_pos_1 = TensLinear(1, dim_v, 1)
    if kind == self.K_DISC:
      self.x_to_E = XToEnergy(dim_a)
  def self_init(self):
    if self.pos_1_mutable:
      with torch.no_grad():
        self.lin_push_pos_1.W.zero_()
  def forward(self, tup):
    # unpack the tuple
    E, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
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
      contexttup, (
        x_a + self.res_embed(metadata),
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
    Δxtup, ytup = self.amino_sub_block((contexttup, (x_a, x_v, x_d), ytup, pos_0_tup, pos_1_tup))
    Δx_a, Δx_v, Δx_d = Δxtup
    x_a = x_a + Δx_a
    x_v = x_v + Δx_v
    x_d = x_d + Δx_d
    # update pos_1
    if self.pos_1_mutable:
      pos_1 = pos_1 + self.lin_push_pos_1(x_v).squeeze(-2)
      pos_1_tup = pos_1, None, None # invalidate previous graph and r_ij
    # update energy for discriminator
    if self.kind == self.K_DISC:
      E = E + self.x_to_E(x_a)
    return E, contexttup, (x_a, x_v, x_d), ytup, pos_0_tup, pos_1_tup

def initialize_tup(self, E, pos_0, pos_1, box, metadata):
  """ Utility function that sets up the tuples that Block operates on.
      self should just be some object containing the correct dim_a, dim_v, dim_d
      values as attributes. head is (t, 0) or None. """
  device = pos_0.device
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
  return E, contexttup, xtup, ytup, pos_0_tup, pos_1_tup

class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.readin_z = TensLinear(1, config["diffusion_steps"], self.dim_v)
    self.blocks = nn.Sequential(*[
      Block(config, Block.K_GEN, None)
      for i in range(config["depth"])
    ])
  def forward(self, pos_0, pos_0_noised, z, box, metadata):
    tup = initialize_tup(self, None, pos_0, pos_0_noised, box, metadata)
    # put z:(batch, atoms, diffusion_steps, 3) into initial activations
    x_v = tup[2][1]
    x_v = x_v + self.readin_z(z)
    tup = tup[:2] + (tup[2][:1] + (x_v,) + tup[2][2:],) + tup[3:] # modify tuple...
    # apply network
    tup = self.blocks(tup)
    head, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    return norm_grad(pos_1_tup[0])


def diffuser_generate_with_epsilon(model, pos_0, metadata, steps:int):
  """ pos_0: (batch, atoms, 3)
      ans: pos_1, pos_0_noised, epsilon
      pos_1, pos_0_noised: (batch, atoms, 3)
      epsilon: (batch, atoms, steps, 3) """
  batch, atoms, must_be[3] = pos_0.shape
  device = pos_0.device
  with torch.no_grad():
    epsilon = []
    pos_0_noised = pos_0 + model.sigma_t(torch.zeros(1, device=device))[:, None, None]*torch.randn_like(pos_0)
    pos_1 = pos_0_noised.clone()
    t_list = np.linspace(0., 1., steps + 1)
    for i in range(steps):
      t = torch.tensor([t_list[i]], device=device, dtype=torch.float32)
      tdec = torch.tensor([t_list[i + 1]], device=device, dtype=torch.float32)
      sigma_t = model.sigma_t(t)[:, None, None]
      sigma_tdec = model.sigma_t(tdec)[:, None, None]
      dsigma = torch.sqrt(sigma_t**2 - sigma_tdec**2)
      pred_noise = pos_1 - model.dn(t, pos_0, pos_1, model.box, metadata)
      pos_1 -= ((dsigma/sigma_t)**2)*pred_noise
      epsilon.append(torch.randn_like(pos_0))
      pos_1 += (dsigma*sigma_tdec/sigma_t)*epsilon[-1]
  return pos_1, pos_0_noised, torch.stack(epsilon, dim=2)


class WGAN3D:
  is_gan = True
  def __init__(self, config:Config):
    self.randgen = TensorRandGen()
    self.config = config
    self.gen = Generator(config).to(config.device)
    self.gen.apply(weights_init)
    assert space_dim(config) == 3
    self.box = config.predictor.get_box()
    self.tensbox = torch.tensor(self.box, dtype=torch.float32, device="cuda")
    self.init_optim()
    self.distill_model = None
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.sched_gen = ErrorTrackingScheduler(get_lr_fn_warmup_and_decay(
      self.config["lr_gen"], 0.1, 128, self.config["gamma_gen"]))
    self.optim_gen = torch.optim.AdamW(get_params_for_optim(self.gen),
      self.sched_gen.lr(), betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    ans = WGAN3D(config)
    ans.gen.load_state_dict(states["gen"])
    return ans
  @staticmethod
  def makenew(config):
    return WGAN3D(config)
  def save_to_dict(self):
    return {
        "gen": self.gen.state_dict(),
      }
  def load_diffuser(self):
    model = load(self.config["distill_model"], override_base=self.config.pred_spec)
    assert hasattr(model, "dn"), "Expected model to be a denoising model (has nn at .dn)"
    assert hasattr(model, "sigma_t"), "Expected model to be a denoising model (has noise schedule at .sigma_t)"
    assert model.box == self.box, f"box mismatch {model.box} != {self.box}"
    # turn off gradients for model params
    for param in model.dn.parameters():
      param.requires_grad = False
    # update our config to save the same sigma_max as the model we depend on
    self.config.arch_specific["sigma_max"] = model.config["sigma_max"]
    return model
  def train_step(self, traj_state):
    """ x: (L, batch, poly_len, 3) """
    if self.distill_model is None:
      self.distill_model = self.load_diffuser()
    x = traj_state.x
    L, batch, atoms, must_be[3] = x.shape
    loss = self.mimic_step(x, traj_state.metadata)
    self.step_count += 1
    self.sched_gen.step(self.step_count, self.optim_gen)
    return loss
  def mimic_step(self, x, metadata):
    L, batch, atoms, must_be[3] = x.shape
    x_0 = x.reshape(L*batch, atoms, 3)
    x_1, x_0_noised, epsilon = diffuser_generate_with_epsilon(self.distill_model, x_0, metadata, self.config["diffusion_steps"])
    x_1_pred = self.gen(x_0, x_0_noised, epsilon, self.box, metadata)
    loss = ((x_1_pred - x_1)**2).sum(-1).mean()
    self.optim_gen.zero_grad()
    loss.backward()
    self.optim_gen.step()
    return loss.item()
  def generate(self, x_0, metadata):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    epsilon = self.randgen.randn(1, (batch, atoms, self.config["diffusion_steps"]))
    x_0_noised = self.config["sigma_max"]*self.randgen.randn(1, (batch, atoms))
    x_1 = self.gen(x_0, x_0_noised, epsilon, self.box, metadata)
    return x_1.reshape(*leading_dims, atoms, 3)
  def predict(self, state:ModelState):
    with torch.no_grad():
      return self.generate(state.x, state.metadata)


class GANTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    loss = self.model.train_step(trajs)
    print(f"{i}\t ℒ = {loss:05.6f}")
    self.board.scalar("loss", i, loss)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer
