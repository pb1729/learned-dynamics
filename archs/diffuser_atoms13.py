from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod, expand_dim
from managan.layers_common import *
from managan.config import Config
from managan.tensor_products import TensLinear, TensConv1d, tens_sigmoid, TensSigmoid, TensGroupNorm, TensorRandGen
from managan.codegen_tensor_products import Cow0, Cow1
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT
from managan.flavour_layers import ResiduesPosEncode, ResidueAtomEmbed, LinAminoToAtomSmall, LinAtomToAminoSmall
from managan.predictor import ModelState


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor, neighbours_max:int=64):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos, neighbours_max=neighbours_max)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij

def make_mlp(dim_in, dim_out=None, dim_hidd=None, leak=0.1):
  if dim_out is None: dim_out = dim_in
  if dim_hidd is None: dim_hidd = max(dim_in, dim_out)
  return nn.Sequential(
      nn.Linear(dim_in, dim_hidd),
      nn.LeakyReLU(leak),
      nn.Linear(dim_hidd, dim_hidd),
      nn.LeakyReLU(leak),
      nn.Linear(dim_hidd, dim_out, bias=False),
    )

def radial_encode(r, n, rmax):
  """ r: (..., 3)
      ans: (..., n) """
  hn = (n + 1)//2
  coeffs = (torch.pi/rmax)*(1 + torch.arange(0, hn, device=r.device)//2)
  dist = torch.linalg.vector_norm(r, dim=-1)[..., None]
  phase = coeffs*dist
  return torch.cat([
    torch.cos(phase),
    torch.sin(phase[..., :(n - hn)])
  ], dim=-1)

def add_tens_prod_submodules(self, dim_a, dim_v, chan):
  self.Y_000 = TensLinear(0, chan, dim_a)
  self.Y_110 = TensLinear(0, chan, dim_a)
  self.Y_011 = TensLinear(1, chan, dim_v)
  self.Y_101 = TensLinear(1, chan, dim_v)
  self.Y_111 = TensLinear(1, chan, dim_v)

def apply_tens_prod_submodules_o0(self, prods):
  y_000, y_110 = prods
  return self.Y_000(y_000) + self.Y_110(y_110)

def apply_tens_prod_submodules_o1(self, prods):
  y_011, y_101, y_111 = prods
  return self.Y_011(y_011) + self.Y_101(y_101) + self.Y_111(y_111)

class SelfTensProds(nn.Module):
  def __init__(self, dim_a, dim_v, chan, groups=8):
    super().__init__()
    # submodules:
    self.L0 = TensLinear(0, dim_a, chan)
    self.R0 = TensLinear(0, dim_a, chan)
    self.L1 = TensLinear(1, dim_v, chan)
    self.R1 = TensLinear(1, dim_v, chan)
    self.update_l_0 = TensLinear(0, dim_a, chan)
    self.update_r_0 = TensLinear(0, dim_a, chan)
    add_tens_prod_submodules(self, dim_a, dim_v, chan)
    self.mlp_a = make_mlp(dim_a)
    # group norms for output
    self.gn_a = TensGroupNorm(0, dim_a, groups)
    self.gn_v = TensGroupNorm(1, dim_v, groups)
  def forward(self, x_a, x_v):
    *rest, dim_a = x_a.shape
    *must_be[rest], dim_v, must_be[3] = x_v.shape
    prod_rest = prod(rest)
    x_a = x_a.reshape(prod_rest, dim_a)
    x_v = x_v.reshape(prod_rest, dim_v, 3)
    # begin computations
    l_0 = self.L0(x_a)
    r_0 = self.R0(x_a)
    l_1 = self.L1(x_v)
    r_1 = self.R1(x_v)
    y_a = apply_tens_prod_submodules_o0(self, Cow0.apply(l_0, l_1, r_0, r_1))
    y_a = y_a + self.mlp_a(y_a)
    l_0 = l_0 + self.update_l_0(y_a)
    r_0 = r_0 + self.update_r_0(y_a)
    y_v = apply_tens_prod_submodules_o1(self, Cow1.apply(l_0, l_1, r_0, r_1))
    # reshape back to original shape
    y_a = y_a.reshape(*rest, dim_a)
    y_v = y_v.reshape(*rest, dim_v, 3)
    return self.gn_a(y_a), self.gn_v(y_v)

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

class DisplacementTensors(nn.Module):
  """ Given node features and positions, extract a neighbour graph and perform ACE-like embedding. """
  def __init__(self, r0:float, dim_a:int, dim_v:int):
    super().__init__()
    self.r0 = r0
    self.dim_a = dim_a
    self.mlp_radial = make_mlp(dim_a, dim_a)
    self.readout_v = TensLinear(1, dim_a, dim_v)
  def forward(self, graph, r_ij, res_emb):
    res_emb_j = edges_read_dst(graph, res_emb)
    rad_enc_ij = self.mlp_radial(radial_encode(r_ij, self.dim_a, self.r0) + res_emb_j)
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                 # (edges, dim_a)
    φ_v_ij = rad_enc_ij[..., None] * r_ij[..., None, :] # (edges, dim_a, 3)
    A_a_i = edges_reduce_src(graph, φ_a_ij) # (batch, nodes, dim_a)
    A_v_i = edges_reduce_src(graph, φ_v_ij) # (batch, nodes, dim_a, 3)
    return A_a_i, self.readout_v(A_v_i)


class Messages(nn.Module):
  def __init__(self, r0:float, dim_a:int, dim_v:int, chan:int):
    super().__init__()
    self.r0 = r0
    self.dim_a = dim_a
    # submodules
    self.L0 = TensLinear(0, dim_a, chan)
    self.L1 = TensLinear(1, dim_v, chan)
    add_tens_prod_submodules(self, dim_a, dim_v, chan)
    self.lin_enc = nn.Linear(dim_a, chan)
    self.mlp_a = make_mlp(dim_a) # MLP that we apply to edge activations
  def forward(self, graph, r_ij, x_a, x_v):
    rad_enc_ij = self.lin_enc(radial_encode(r_ij, self.dim_a, self.r0))
    r_ij = tens_sigmoid(1, r_ij*(7./self.r0)) # normalize radial separations
    φ_a_ij = rad_enc_ij                                 # (edges, chan)
    φ_v_ij = rad_enc_ij[..., None] * r_ij[..., None, :] # (edges, chan, 3)
    l_a_j = edges_read_dst(graph, self.L0(x_a)) # (edges, chan)
    l_v_j = edges_read_dst(graph, self.L1(x_v)) # (edges, chan, 3)
    ψ_a_ij = apply_tens_prod_submodules_o0(self, Cow0.apply(l_a_j, l_v_j, φ_a_ij, φ_v_ij))
    ψ_v_ij = apply_tens_prod_submodules_o1(self, Cow1.apply(l_a_j, l_v_j, φ_a_ij, φ_v_ij))
    ψ_a_ij = ψ_a_ij + self.mlp_a(ψ_a_ij)
    B_a_i = edges_reduce_src(graph, ψ_a_ij) # (batch, nodes, dim_a)
    B_v_i = edges_reduce_src(graph, ψ_v_ij) # (batch, nodes, dim_v, 3)
    return 0.1*B_a_i, 0.1*B_v_i

class AminosConv(nn.Module):
  def __init__(self, config):
    super().__init__()
    dim_a, dim_v = config["dim_a"], config["dim_v"]
    self.readin_a = LinAtomToAminoSmall(0, dim_a, dim_a, dim_a)
    self.readin_v = LinAtomToAminoSmall(1, dim_v, dim_v, dim_a)
    self.convs_a = nn.Sequential(
      TensConv1d(0, dim_a, 3),
      nn.LeakyReLU(0.1),
      TensConv1d(0, dim_a, 3))
    self.convs_v = TensConv1d(1, dim_v, 5)
    self.readout_a = LinAminoToAtomSmall(0, dim_a, dim_a, dim_a)
    self.readout_v = LinAminoToAtomSmall(1, dim_v, dim_v, dim_a)
  def forward(self, xtup, metadata, res_emb):
    x_a, x_v = xtup # per atom activations
    # readin
    y_a = self.readin_a(x_a, res_emb, metadata)
    y_v = self.readin_v(x_v, res_emb, metadata)
    # 1d conv along chain
    y_a = self.convs_a(y_a)
    y_v = self.convs_v(y_v)
    # readout
    Δxtup = (
      0.2*self.readout_a(y_a, res_emb, metadata),
      0.2*self.readout_v(y_v, res_emb, metadata))
    return Δxtup


class Block(nn.Module):
  """ Processing block for nets. """
  def __init__(self, config:Config, pos_1_mutable=False):
    super().__init__()
    dim_a, dim_v, chan = config["dim_a"], config["dim_v"], config["chan"]
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
    self.disptens_0 = DisplacementTensors(config["r_cut"], dim_a, dim_v)
    self.disptens_1 = DisplacementTensors(config["r_cut"], dim_a, dim_v)
    self.res_pos_encode_0 = ResiduesPosEncode(dim_v)
    self.res_pos_encode_1 = ResiduesPosEncode(dim_v)
    self.tprods_self = SelfTensProds(dim_a, dim_v, chan)
    self.aminos_conv = AminosConv(config)
    self.messages_0 = Messages(config["r_cut"], dim_a, dim_v, chan)
    self.messages_1 = Messages(config["r_cut"], dim_a, dim_v, chan)
    if pos_1_mutable:
      self.lin_push_pos_1 = TensLinear(1, dim_v, 1)
  def self_init(self):
    if self.pos_1_mutable:
      with torch.no_grad():
        self.lin_push_pos_1.W.zero_()
  def forward(self, tup):
    # unpack the tuple
    t, contexttup, xtup, pos_0_tup, pos_1_tup = tup
    box, metadata = contexttup
    x_a, x_v = xtup # per atom activations
    pos_0, graph_0, r_ij_0 = pos_0_tup
    pos_1, graph_1, r_ij_1 = pos_1_tup
    # graph setup
    if graph_0 is None: # pos_0 was modified since graph was last computed
      graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0, neighbours_max=self.neighbours_max)
      pos_0_tup = pos_0, graph_0, r_ij_0
    if graph_1 is None: # pos_1 was modified since graph was last computed
      graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1, neighbours_max=self.neighbours_max)
      pos_1_tup = pos_1, graph_1, r_ij_1
    # embeddings for residues / atoms in residue
    res_emb_a = self.res_embed(metadata)[None]
    # ACE subblock
    res_emb_expanded = expand_dim(res_emb_a, 0, pos_0.shape[0])
    Δx_a_0, Δx_v_0 = self.disptens_0(graph_0, r_ij_0, res_emb_expanded)
    Δx_a_1, Δx_v_1 = self.disptens_1(graph_1, r_ij_1, res_emb_expanded)
    x_a = x_a + Δx_a_0 + Δx_a_1
    x_v = x_v + Δx_v_0 + Δx_v_1
    # Local tensor products and MLP
    Δx_a, Δx_v = self.tprods_self(
      x_a + res_emb_a + self.embed_t(t)[:, None],
      x_v + self.pos_embed(pos_0, pos_1) + 0.2*(self.res_pos_encode_0(pos_0, metadata) + self.res_pos_encode_1(pos_1, metadata)),
    )
    x_a = x_a + Δx_a
    x_v = x_v + Δx_v
    # aminos conv
    Δx_a, Δx_v = self.aminos_conv((x_a, x_v), metadata, res_emb_a)
    x_a = x_a + Δx_a
    x_v = x_v + Δx_v
    # messages
    Δx_a_0, Δx_v_0 = self.messages_0(graph_0, r_ij_0, x_a, x_v)
    Δx_a_1, Δx_v_1 = self.messages_1(graph_1, r_ij_1, x_a, x_v)
    x_a = x_a + Δx_a_0 + Δx_a_1
    x_v = x_v + Δx_v_0 + Δx_v_1
    # update pos_1
    if self.pos_1_mutable:
      pos_1 = pos_1 + self.lin_push_pos_1(x_v).squeeze(-2)
      pos_1_tup = pos_1, None, None # invalidate previous graph and r_ij
    return t, contexttup, (x_a, x_v), pos_0_tup, pos_1_tup


def initialize_tup(self, head, pos_0, pos_1, box, metadata):
  """ Utility function that sets up the tuples that Block operates on.
      self should just be some object containing the correct dim_a, dim_v, dim_d
      values as attributes. head is t or None. """
  device = pos_0.device
  batch,          atoms,          must_be[3] = pos_0.shape
  must_be[batch], must_be[atoms], must_be[3] = pos_1.shape
  aminos = len(metadata.seq)
  pos_0, pos_1 = pos_0.contiguous(), pos_1.contiguous()
  # run the main network
  contexttup = box, metadata
  xtup = (
    torch.zeros(batch, atoms, self.dim_a, device=device),
    torch.zeros(batch, atoms, self.dim_v, 3, device=device))
  pos_0_tup = pos_0, None, None
  pos_1_tup = pos_1, None, None
  return head, contexttup, xtup, pos_0_tup, pos_1_tup


class Denoiser(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dim_a, self.dim_v = config["dim_a"], config["dim_v"]
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config, pos_1_mutable=True)
      for i in range(config["depth"])
    ])
    self.dpos_readout = TensLinear(1, 1, self.dim_v)
    self.readout_tmod = TimeLinearModulation(config["t_embed_hdim"], 1, self.dim_v)
    self.readout_lin = TensLinear(1, self.dim_v, 1)
  def forward(self, t, pos_0, pos_1, box, metadata):
    tup = initialize_tup(self, t, pos_0, pos_1, box, metadata)
    tup = self.blocks(tup)
    t, contexttup, xtup, pos_0_tup, pos_1_tup = tup
    dpos = (pos_1 - pos_1_tup[0])[..., None, :]
    return self.readout_lin(
      self.readout_tmod(xtup[1] + self.dpos_readout(dpos), t)
    ).squeeze(-2)



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
    epsilon_pred = self.dn(t, x_0, x_1_noised, self.box, metadata)
    loss = (torch.tanh(3.*sigma)*(epsilon_pred - epsilon)**2).mean()
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return loss.item()
  def generate(self, x_0, metadata, steps=32):
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
      epsilon_pred = self.dn(t, x_0, ans, self.box, metadata)
      pred_noise = epsilon_pred*dsigma
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
