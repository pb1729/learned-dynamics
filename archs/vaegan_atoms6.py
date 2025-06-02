from typing_extensions import Tuple, List, Optional, Union

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
from managan.learn_common import ErrorTrackingScheduler, get_lr_fn_warmup_and_decay, get_endpt_pen, norm_grad
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


class VAEReadout0(nn.Module):
  def __init__(self, dim_a:int, zdim:int):
    super().__init__()
    self.layers_mu_a = nn.Sequential(
      MLP(dim_a),
      nn.Linear(dim_a, zdim, bias=False))
    self.layers_trans_a = nn.Sequential(
      MLP(dim_a),
      nn.Linear(dim_a, zdim, bias=False))
  def forward(self, x_a, x_v, x_d):
    mu = self.layers_mu_a(x_a)
    trans = self.layers_trans_a(x_a)
    return mu, trans

class VAEReadout1(nn.Module):
  def __init__(self, dim_a:int, dim_v:int, dim_d:int, zdim:int):
    super().__init__()
    self.layers_mu_v = nn.Sequential(
      VectorMLP(dim_v),
      TensLinear(1, dim_v, zdim))
    self.layers_trans_a = nn.Sequential(
      MLP(dim_a),
      nn.Linear(dim_a, zdim, bias=False))
    self.layers_trans_d = TensLinear(2, dim_d, zdim)
  def forward(self, x_a, x_v, x_d):
    mu = self.layers_mu_v(x_v)
    trans = (self.layers_trans_a(x_a) + 1)[..., None, None]*torch.eye(3, device=x_a.device) + self.layers_trans_d(x_d)
    return mu, trans

class GaussELBORandgen:
  def __init__(self, base_randgen:TensorRandGen, calls:List[Tuple[int, int]]):
    self.base_randgen = base_randgen
    self.calls = calls
    self.i:Optional[int] = None # current index into calls
    self.z:Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
  def set_transform(self, transform, seed=None):
    self.base_randgen.set_transform(transform, seed)
  def clear_transform(self):
    self.base_randgen.clear_transform()
  def open(self, z=None):
    assert self.i is None, "illegal double open"
    self.i = 0
    self.z = z
    if z is not None:
      batch, atoms = z[0][0].shape[0:2] # peek at the first mu to determine batch and atom count
      for (inds, chan), (mu, trans) in zip(self.calls, z): # check that the shapes are correct
        must_be[batch], must_be[atoms], must_be[chan], *must_be[(3,)*inds] = mu.shape
        must_be[batch], must_be[atoms], must_be[chan], *must_be[(3,)*(2*inds)] = trans.shape
  def close(self):
    assert self.i == len(self.calls), "GaussELBORandgen closed before all calls were made"
    self.i = None
    self.z = None
  def get_elbo_loss(self):
    assert self.z is not None, "Can't compute ELBO loss while in \"prior\" mode."
    ans = 0.
    for (inds, chan), (mu, trans) in zip(self.calls, self.z):
      if inds == 0:
        cov = trans*trans
        ans += 0.5*(cov + mu*mu - torch.log(cov))
      elif inds == 1:
        cov = torch.einsum("...ik,...jk->...ij", trans, trans)
        ans += 0.5*(torch.einsum("...ii->...", cov) + (mu*mu).sum(-1) - torch.log(torch.det(trans)**2 + 1e-6))
      else:
        assert False, f"Can't compute ELBO loss for {inds} indices."
    return ans
  def randn(self, spatial_indices:int, shape):
    assert spatial_indices in [0, 1], "only 0 index and 1 index cases are supported right now!"
    assert self.i is not None, "must call open() before we can use the GaussELBORandgen"
    inds, chan = self.calls[self.i]
    batch, atoms, must_be[chan] = shape
    if self.z is None:
      ans = self.base_randgen.randn(spatial_indices, shape)
    else:
      mu, trans = self.z[self.i]
      must_be[batch], must_be[atoms], must_be[chan], *must_be[(3,)*inds] = mu.shape
      must_be[batch], must_be[atoms], must_be[chan], *must_be[(3,)*(2*inds)] = trans.shape
      epsilon = self.base_randgen.randn(spatial_indices, shape)
      if inds == 0:
        ans = mu + trans*epsilon
      else: # inds == 1
        ans = mu + torch.einsum('...i,...ij->...j', epsilon, trans)
    self.i += 1
    return ans


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
  K_ENC = 2
  def __init__(self, config:Config, kind:int, randgen:Union[None, TensorRandGen, GaussELBORandgen]):
    super().__init__()
    dim_a, dim_v, dim_d, chan = config["dim_a"], config["dim_v"], config["dim_d"], config["chan"]
    self.zdim_0, self.zdim_1 = config["zdim_0"], config["zdim_1"]
    self.kind:int = kind
    self.pos_1_mutable:bool = (kind == self.K_GEN)
    self.r0:float = config["r_cut"] # cutoff radius for atom interactions
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
    if kind == self.K_GEN:
      self.rand_lin_a = TensLinear(0, config["zdim_0"], dim_a)
      self.rand_lin_v = TensLinear(1, config["zdim_1"], dim_v)
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
      graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0)
      pos_0_tup = pos_0, graph_0, r_ij_0
    if graph_1 is None: # pos_1 was modified since graph was last computed
      graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1)
      pos_1_tup = pos_1, graph_1, r_ij_1
    # add noise to activations for generator
    if self.kind == self.K_GEN:
      x_a = x_a + self.rand_lin_a(self.randgen.randn(0, x_a.shape[:-1] + (self.zdim_0,)))
      x_v = x_v + self.rand_lin_v(self.randgen.randn(1, x_v.shape[:-2] + (self.zdim_1,)))
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

def initialize_tup(self, head, pos_0, pos_1, box, metadata):
  """ Utility function that sets up the tuples that Block operates on.
      self should just be some object containing the correct dim_a, dim_v, dim_d
      values as attributes. head is t or 0 or None. """
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
  return head, contexttup, xtup, ytup, pos_0_tup, pos_1_tup

class Discriminator(nn.Module):
  def __init__(self, config, randgen:TensorRandGen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, Block.K_DISC, randgen)
      for i in range(config["depth"])
    ])
  def forward(self, pos_0, pos_1, box, metadata):
    tup = initialize_tup(self, 0., pos_0, pos_1, box, metadata)
    tup = self.blocks(tup)
    E, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    return E.mean(-1) # average over atoms

class Generator(nn.Module):
  def __init__(self, config, randgen:GaussELBORandgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.randgen = randgen
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, Block.K_GEN, randgen)
      for i in range(config["depth"])
    ])
  def forward(self, pos_0, box, metadata):
    tup = initialize_tup(self, 0., pos_0, pos_0, box, metadata)
    tup = self.blocks(tup)
    E, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    return norm_grad(pos_1_tup[0])

class Encoder(nn.Module):
  @staticmethod
  def _get_calls(config):
    ans = []
    for i in range(config["depth"]):
      ans.append((0, config["zdim_0"]))
      ans.append((1, config["zdim_1"]))
    return ans
  def __init__(self, config, randgen:TensorRandGen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.calls = self._get_calls(config)
    self.randgen = None
    self.elbo_randgen = GaussELBORandgen(randgen, self.calls)
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, Block.K_ENC, self.randgen)
      for i in range(config["depth"])
    ])
    self.readouts = nn.ModuleList([
      VAEReadout0(self.dim_a, chan)
      if inds == 0 else
      VAEReadout1(self.dim_a, self.dim_v, self.dim_d, chan)
      for inds, chan in self.calls
    ])
  def forward(self, pos_0, pos_1, box, metadata):
    tup = initialize_tup(self, 0., pos_0, pos_1, box, metadata)
    tup = self.blocks(tup)
    E, contexttup, xtup, ytup, pos_0_tup, pos_1_tup = tup
    z = [
      readout_module(*xtup)
      for readout_module in self.readouts
    ]
    return z

class Decoder(nn.Module):
  def __init__(self, config, randgen:GaussELBORandgen):
    super().__init__()
    self.elbo_randgen = randgen
    self.gen = Generator(config, randgen)
  def forward(self, pos_0, z, box, metadata):
    self.elbo_randgen.open(z)
    ans = self.gen(pos_0, box, metadata)
    loss_elbo = self.elbo_randgen.get_elbo_loss()
    self.elbo_randgen.close()
    return ans, loss_elbo
  def prior_gen(self, pos_0, box, metadata):
    self.elbo_randgen.open(None)
    ans = self.gen(pos_0, box, metadata)
    self.elbo_randgen.close()
    return ans


class WGAN3D:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.config = config
    self.discs = []
    for _ in range(config["ndiscs"]):
      self.add_new_disc()
    self.enc = Encoder(config, self.randgen).to(config.device)
    self.enc.apply(weights_init)
    self.dec = Decoder(config, self.enc.elbo_randgen).to(config.device)
    self.dec.apply(weights_init)
    assert space_dim(config) == 3
    self.box = config.predictor.get_box()
    self.tensbox = torch.tensor(self.box, dtype=torch.float32, device="cuda")
    self.init_optim()
  def add_new_disc(self):
    self.discs.append(Discriminator(self.config, self.randgen).to(self.config.device))
    self.discs[-1].apply(weights_init)
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.sched_disc = ErrorTrackingScheduler(get_lr_fn_warmup_and_decay(
      self.config["lr_disc"], 0.1, 128, self.config["gamma_disc"]))
    self.optim_disc = torch.optim.AdamW(get_params_for_optim(*self.discs),
      self.sched_disc.lr(), betas, weight_decay=self.config["weight_decay"])
    self.sched_enc = ErrorTrackingScheduler(get_lr_fn_warmup_and_decay(
      self.config["lr_enc"], 0.1, 128, self.config["gamma_enc"]))
    self.optim_enc = torch.optim.AdamW(get_params_for_optim(self.enc),
      self.sched_enc.lr(), betas, weight_decay=self.config["weight_decay"])
    self.sched_dec = ErrorTrackingScheduler(get_lr_fn_warmup_and_decay(
      self.config["lr_dec"], 0.1, 128, self.config["gamma_dec"]))
    self.optim_dec = torch.optim.AdamW(get_params_for_optim(self.dec),
      self.sched_dec.lr(), betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    ans = WGAN3D(config)
    for disc, state_dict in zip(ans.discs, states["discs"]):
      disc.load_state_dict(state_dict)
    ans.enc.load_state_dict(states["enc"])
    ans.dec.load_state_dict(states["dec"])
    return ans
  @staticmethod
  def makenew(config):
    return WGAN3D(config)
  def save_to_dict(self):
    return {
        "discs": [disc.state_dict() for disc in self.discs],
        "enc": self.enc.state_dict(),
        "dec": self.dec.state_dict(),
      }
  def train_step(self, traj_state):
    """ x: (L, batch, poly_len, 3) """
    x = traj_state.x
    L, batch, atoms, must_be[3] = x.shape
    loss_d = self.discs_step(x, traj_state.metadata)
    if "gen_train" in self.config and not self.config["gen_train"]:
      loss_g, loss_a, loss_e = 0., 0., 0.
    else:
      loss_g = self.gen_step(x, traj_state.metadata)
      loss_a, loss_e = self.autoenc_step(x, traj_state.metadata)
    self.step_count += 1
    self.sched_disc.step(self.step_count, self.optim_disc)
    self.sched_enc.step(self.step_count, self.optim_enc)
    self.sched_dec.step(self.step_count, self.optim_dec)
    return loss_d, loss_g, loss_a, loss_e
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
    self.optim_disc.zero_grad()
    loss.backward()
    self.optim_disc.step()
    return loss.item()
  def disc_loss(self, disc, x_0, x_r, x_g, metadata):
    batch,          atoms,          must_be[3] = x_0.shape
    must_be[batch], must_be[atoms], must_be[3] = x_r.shape
    must_be[batch], must_be[atoms], must_be[3] = x_g.shape
    # train on real data
    y_r = disc(x_0, x_r, self.box, metadata) # (batch,)
    # train on generated data
    y_g = disc(x_0, x_g, self.box, metadata) # (batch,)
    # endpoint penalty on interpolated data
    endpt_pen = get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g, self.box, metadata, self.config["lambda_wass"], self.config["lambda_l2"])
    # overall loss
    if self.config["hinge"]:
      loss = torch.relu(1. + y_r).mean() + torch.relu(1. - y_g).mean() + self.config["hinge_leak"]*(y_r.mean() - y_g.mean())
    else:
      loss = y_r.mean() - y_g.mean()
    return loss + endpt_pen
  def gen_step(self, x, metadata):
    *_, atoms, must_be[3] = x.shape
    x_g = x
    loss_g = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:-1], metadata)
      x_0 = x[:-nsteps]
      y_g = disc(x_0.reshape(-1, atoms, 3), x_g.reshape(-1, atoms, 3), self.box, metadata)
      loss_g = loss_g + y_g.mean()
    # backprop, update
    self.optim_dec.zero_grad() # carry over to autoenc_step
    loss_g.backward()
    return loss_g.item()
  def autoenc_step(self, x, metadata):
    L, batch, atoms, must_be[3] = x.shape
    x_0, x_1 = x[:-1].reshape((L - 1)*batch, atoms, 3), x[1:].reshape((L - 1)*batch, atoms, 3)
    z = self.enc(x_0, x_1, self.box, metadata)
    x_dec, loss_elbo = self.dec(x_0, z, self.box, metadata)
    lambda_rec = self.config["lambda_rec"] if "lambda_rec" in self.config else 1.0
    loss_recons = lambda_rec*((x_dec - x_1)**2).sum(-1).mean()
    loss_elbo = self.config["lambda_elbo"]*loss_elbo.mean()
    loss = loss_recons + loss_elbo
    self.optim_enc.zero_grad()
    loss.backward()
    self.optim_enc.step()
    self.optim_dec.step() # carry over from gen_step
    return loss_recons.item(), loss_elbo.item()
  def generate(self, x_0, metadata):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    ans = self.dec.prior_gen(x_0, self.box, metadata)
    return ans.reshape(*leading_dims, atoms, 3)
  def predict(self, state:ModelState):
    with torch.no_grad():
      return self.generate(state.x, state.metadata)


class GANTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    loss_d, loss_g, loss_a, loss_e = self.model.train_step(trajs)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}   \t ℒᴳ = {loss_g:05.6f}  \t ℒᴬ = {loss_a:05.6f}  \t ℒᴱ = {loss_e:05.6f}")
    self.board.scalar("loss_d", i, loss_d)
    self.board.scalar("loss_g", i, loss_g)
    self.board.scalar("loss_a", i, loss_a)
    self.board.scalar("loss_e", i, loss_e)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer


