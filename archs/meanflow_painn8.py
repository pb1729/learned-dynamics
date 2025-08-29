from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.utils import must_be, prod
from managan.layers_common import *
from managan.config import Config
from managan.tensor_products import *
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap
from managan.grouping import DEFAULT, get_params_for_optim
from managan.flavour_layers import ResidueAtomEmbed, get_bond_graph
from managan.predictor import ModelState
from managan.sim_utils import OpenMMMetadata


# Implementation of the PAINN from "Implicit Transfer Operator" by Schreiner et al 2023
# This initial version is fixed lag, though the version in the paper allows variable lag.
# Meanflow: https://arxiv.org/pdf/2505.13447


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


class MLP(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(dim_in, dim_out),
      nn.SiLU(),
      nn.Linear(dim_out, dim_out),
      nn.SiLU(),
      nn.Linear(dim_out, dim_out),
    )
  def forward(self, x_a):
    """ x_a: (..., dim_in)
        ans: (..., dim_out) """
    return self.layers(x_a)

class Update(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.n = n
    self.lin_l = TensLinear(1, n, n)
    self.lin_r = TensLinear(1, n, n)
    self.lin_v = TensLinear(1, n, n)
    self.mlp = MLP(2*n, 3*n)
    assert n % 4 == 0
    self.gn = ScalGroupNorm(n, 4, epsilon=1.)
  def forward(self, x_a, x_v):
    """ x_a: (batch, atoms, n)
        x_v: (batch, atoms, n, 3) """
    x_v_l = self.lin_l(x_v)
    x_v_r = self.lin_r(x_v)
    inner_prod = (x_v_l*x_v_r).sum(-1) # (batch, atoms, n)
    y = self.mlp(torch.cat([x_a, torch.linalg.vector_norm(x_v_r, dim=-1)], dim=-1)) # (batch, atoms, 3n)
    x_v = self.lin_v(x_v)
    x_v = x_v*y[..., :self.n, None]
    inner_prod = self.gn(inner_prod*y[..., self.n:2*self.n])
    x_a = inner_prod + y[..., 2*self.n:]
    x_v = x_v/torch.sqrt(1. + (x_v**2).sum(-1, keepdim=True))
    return x_a, x_v

class Message(nn.Module):
  def __init__(self, r0:float, n:int):
    super().__init__()
    self.r0 = r0
    self.n = n
    self.epsilon = 0.1 # maybe this should be a parameter later?
    self.mlp_dist = MLP(n, 4*n)
    self.mlp_actv = MLP(n, 4*n)
  def forward(self, graph:Graph, x_a:torch.Tensor, x_v:torch.Tensor, r_ij:torch.Tensor):
    """ x_a: (batch, atoms, n)
        x_v: (batch, atoms, n, 3)
        r_ij: (edges, 3) """
    dist_x_a = self.mlp_dist(radial_encode(r_ij, self.n, self.r0)) # (edges, 4n)
    actv_x_a = self.mlp_actv(edges_read_dst(graph, x_a)) # (edges, 4n)
    prod_x_a = dist_x_a*actv_x_a # (edges, 4n)
    direction = r_ij/torch.sqrt(self.epsilon + (r_ij**2).sum(-1, keepdim=True)) # (edges, 3)
    x_v = edges_read_dst(graph, x_v) # (edges, n, 3)
    cross = torch.linalg.cross(x_v, direction[:, None])
    x_v = x_v*prod_x_a[..., :self.n, None]
    x_v = x_v + cross*prod_x_a[..., self.n:2*self.n, None]
    x_v = x_v + direction[:, None]*prod_x_a[..., 2*self.n:3*self.n, None]
    x_v = edges_reduce_src(graph, x_v)
    x_a = edges_reduce_src(graph, prod_x_a[..., 3*self.n:])
    return x_a, x_v

class Block(nn.Module):
  def __init__(self, r0:float, n:int, pos_mutable:bool=False):
    super().__init__()
    self.pos_mutable = pos_mutable
    self.r0 = r0
    self.message = Message(r0, n)
    self.message_bond = Message(r0, n)
    self.update = Update(n)
    if self.pos_mutable:
      self.lin_push_pos = TensLinear(1, n, 1)
  def forward(self, tup):
    pos, graph, r_ij, graph_bond, r_ij_bond, x_a, x_v, box = tup
    if graph is None: # pos was modified since graph was last computed
      graph, r_ij = graph_setup(self.r0, box, pos)
    dx_a, dx_v = self.message_bond(graph_bond, x_a, x_v, r_ij_bond)
    x_a, x_v = x_a + dx_a, x_v + dx_v
    dx_a, dx_v = self.message(graph, x_a, x_v, r_ij)
    x_a, x_v = x_a + dx_a, x_v + dx_v
    dx_a, dx_v = self.update(x_a, x_v)
    x_a, x_v = x_a + dx_a, x_v + dx_v
    if self.pos_mutable:
      pos = pos + self.lin_push_pos(x_v).squeeze(-2)
      graph, r_ij = None, None # invalidate previous graph and r_ij
    return pos, graph, r_ij, graph_bond, r_ij_bond, x_a, x_v, box


class CPaiNN(nn.Module):
  def __init__(self, r0:float, n:int, blks:int, mutate_pos:bool=False):
    super().__init__()
    self.atom_embed = ResidueAtomEmbed(n)
    self.blocks = nn.Sequential(*[
      Block(r0, n, pos_mutable=mutate_pos)
      for i in range(blks)
    ])
  def forward(self,
      pos:torch.Tensor, graph_bond:Graph, r_ij_bond:torch.Tensor,
      x_a:torch.Tensor, x_v:torch.Tensor,
      box, metadata:OpenMMMetadata):
    x_a = x_a + self.atom_embed(metadata)[None] # add atom embeddings
    pos, graph, r_ij, graph_bond, r_ij_bond, x_a, x_v, box = self.blocks((pos, None, None, graph_bond, r_ij_bond, x_a, x_v, box))
    # Note: some final processing layers removed
    return pos, x_a, x_v

class TimeEmbedding(nn.Module):
  def __init__(self, freqs, outdim):
    super(TimeEmbedding, self).__init__()
    self.freqs = freqs
    self.lin1 = nn.Linear(4*self.freqs, outdim)
  def raw_t_embed(self, st):
    """ st: (batch), (batch) """
    s, t = st
    ang_freqs = torch.exp(-torch.arange(self.freqs, device=s.device)/(self.freqs - 1))
    phases = torch.cat([
      t[:, None] * ang_freqs[None, :],
      (t - s)[:, None] * ang_freqs[None, :]
    ], dim=1)
    return torch.cat([
      torch.sin(phases),
      torch.cos(phases),
    ], dim=1)
  def forward(self, st):
    """ st: (batch), (batch)
        ans: (batch, outdim) """
    return self.lin1(self.raw_t_embed(st))

def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij

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

class SE3ITO(nn.Module):
  def __init__(self, r0:float, n:int, natoms:int, blks_0:int=2, blks_1:int=5):
    super().__init__()
    self.r0 = r0
    self.atom_embeddings = nn.Parameter(torch.randn(natoms, n))
    self.t_embed = TimeEmbedding(32, n)
    self.mlp_embed = MLP(n, n)
    self.mlp_denoise = MLP(n, n)
    self.pos_emb = PosEmbed(n)
    self.cpainn_0 = CPaiNN(r0, n, blks_0)
    self.cpainn_1 = CPaiNN(r0, n, blks_1)
    self.v_readin = TensLinear(1, 1, n)
    self.readout = TensLinear(1, n, 1)
  def get_v_pred(self, st:torch.Tensor, v_pred:torch.Tensor, x1_s_pred:torch.Tensor, x1_t:torch.Tensor):
    s, t = st
    return (1. - (t - s)[:, None, None])*v_pred + x1_t - x1_s_pred
  def forward(self, st:torch.Tensor, x0:torch.Tensor, x1:torch.Tensor, box, metadata:OpenMMMetadata):
    """ st: (batch), (batch)
        x0: (batch, atoms, 3)
        x1: (batch, atoms, 3) """
    x0, x1 = x0.contiguous(), x1.contiguous()
    x1_t = x1 # save for later
    graph_bond = get_bond_graph(x0.shape[0], metadata, x0.device)
    r_ij_bond_0 = x0.reshape(-1, 3)[graph_bond.dst] - x0.reshape(-1, 3)[graph_bond.src]
    r_ij_bond_1 = x1.reshape(-1, 3)[graph_bond.dst] - x1.reshape(-1, 3)[graph_bond.src]
    x_a = self.atom_embeddings[None].expand(x0.shape[0], -1, -1)
    x_v = self.v_readin((x1 - x0)[..., None, :])
    x_a = self.mlp_embed(x_a)
    x0, x_a, x_v = self.cpainn_0(x0, graph_bond, r_ij_bond_0, x_a, x_v, box, metadata)
    x_a = self.mlp_denoise(x_a + self.t_embed(st)[:, None])
    x_v = x_v + self.pos_emb(x0, x1)
    x1, x_a, x_v = self.cpainn_1(x1, graph_bond, r_ij_bond_1, x_a, x_v, box, metadata)
    return self.get_v_pred(st, self.readout(x_v).squeeze(-2), x1, x1_t)





def nodecay_cosine_schedule(t, sigma_max):
  return torch.cos(0.5*torch.pi*t)/torch.sqrt(sigma_max**-2 + torch.sin(0.5*torch.pi*t)**2)


class Meanflow:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.config = config
    self.dn = SE3ITO(config["r0"], config["n"], config["natoms"]).to(config.device)
    #self.dn.apply(weights_init) # TODO: determine if this is necessary!
    self.box = config.predictor.get_box()
    self.tensbox = torch.tensor(self.box, dtype=torch.float32, device="cuda")
    self.sigma_0 = config["sigma_0"]
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim = torch.optim.AdamW(self.dn.parameters(),
      self.config["lr"], betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    ans = Meanflow(config)
    ans.dn.load_state_dict(states["dn"])
    return ans
  @staticmethod
  def makenew(config):
    return Meanflow(config)
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
  def _get_pos_and_vel(self, z_0, z_1, t):
    t = t[:, None, None]
    z_t = t*z_1 + (1 - t)*z_0
    v = z_1 - z_0
    return z_t, v
  def _get_jvp(self, x_prev, z_t, v, s, t, metadata):
    f = lambda z_t, t: self.dn((s, t), x_prev, z_t, self.box, metadata)
    _, jvp = torch.autograd.functional.jvp(
      f,
      (z_t, t),
      (v, torch.ones_like(t)),
      create_graph=True,
    )
    return jvp
  def _diffuser_step(self, x, metadata):
    """ x: (L, batch, poly_len, 3) """
    L, batch, atoms, must_be[3] = x.shape
    x_prev = x[:-1].reshape((L - 1)*batch, atoms, 3)
    z_0 = x[1:].reshape((L - 1)*batch, atoms, 3)
    z_1 = x_prev + self.sigma_0*torch.randn_like(x_prev)
    t = torch.rand((L - 1)*batch, device=x.device)
    s = t*(1 - torch.relu(4*torch.rand((L - 1)*batch, device=x.device) - 3))
    z_t, v = self._get_pos_and_vel(z_0, z_1, t)
    v_target = (v - (t - s)[:, None, None]*self._get_jvp(x_prev, z_t, v, s, t, metadata))
    v_pred = self.dn((s, t), x_prev, z_t, self.box, metadata)
    Δ_sq = ((v_pred - v_target)**2).mean((1, 2))
    if self.config["loss_wt"] == "ph":
      Δ_sq = Δ_sq/(0.5 + Δ_sq.detach())**0.7 # apply pseudo-huber-like weighting, with c=0.5, p=0.7
    loss = Δ_sq.mean()
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return loss.item()
  def generate(self, x_0, metadata):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    z_1 = x_0 + self.sigma_0*torch.randn_like(x_0)
    st = torch.zeros(batch, device=x_0.device), torch.ones(batch, device=x_0.device) # (0, 1)
    v_pred = self.dn(st, x_0, z_1, self.box, metadata)
    ans = z_1 - v_pred
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
    print(f"{i}\t ℒ = {loss:05.6f}")
    self.board.scalar("loss", i, loss)


# export model class and trainer class:
modelclass   = Meanflow
trainerclass = DiffusionDenoiserTrainer
