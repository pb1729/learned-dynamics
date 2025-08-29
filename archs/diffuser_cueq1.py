from typing_extensions import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from managan.polymer_util import space_dim
from managan.utils import must_be, prod
from managan.config import Config
from managan.jacobi_radenc import radial_encode_8
from managan.graph_layers import Graph, edges_read, edges_read_dst, edges_reduce_src, boxwrap, expand_graph
from managan.predictor import ModelState
from managan.cueq_tensor_products import ElementwiseTensorProd_o0, ElementwiseTensorProd_o1, MessageContract
from managan.seq2pdbchain.amino_data import letter_code
from managan.openmm_sims import OpenMMMetadata
from managan.layers_common import weights_init


def graph_setup(r0:float, box:Tuple[float, float, float], pos:torch.Tensor, neighbours_max:int=64):
  tensbox = torch.tensor(box, device="cuda")
  graph = Graph.radius_graph(r0, box, pos, neighbours_max=neighbours_max)
  pos_src, pos_dst = edges_read(graph, pos)
  r_ij = boxwrap(tensbox, pos_dst - pos_src)
  return graph, r_ij


class _VectorSigmoidCh(torch.autograd.Function):
  """ vector activation function with a sigmoid shape in any given direction """
  @staticmethod
  def forward(ctx, v):
    """ x: (..., 3, chan) """
    ctx.save_for_backward(v)
    vv = (v**2).sum(-2, keepdim=True)
    return v/torch.sqrt(1. + vv)
  @staticmethod
  def backward(ctx, grad_output):
    v, = ctx.saved_tensors
    vv = (v**2).sum(-2, keepdim=True)
    quad = 1. + vv
    return (quad*grad_output - (v*grad_output).sum(-2, keepdim=True)*v)*(quad**-1.5)
vector_sigmoid = _VectorSigmoidCh.apply


class GatedLinear(nn.Module):
  def __init__(self, chan):
    super().__init__()
    self.lin_in = nn.Linear(chan, chan)
    self.lin_out = nn.Linear(chan, chan)
    self.lin_gate = nn.Linear(chan, chan)
  def forward(self, x, enc):
    """ x, enc: (..., chan) """
    return self.lin_out(self.lin_in(x)*self.lin_gate(enc))

class VecGatedLinear(nn.Module):
  def __init__(self, chan_in, chan_out, chan_enc, chan_middle):
    super().__init__()
    self.lin_in = nn.Linear(chan_in, chan_middle, bias=False)
    self.lin_out = nn.Linear(chan_middle, chan_out, bias=False)
    self.lin_gate = nn.Linear(chan_enc, chan_middle)
  def forward(self, x, enc):
    """ x: (..., irrep_dim, chan)
        enc: (..., chan) """
    return self.lin_out(self.lin_in(x)*self.lin_gate(enc)[..., None, :])

def make_mlp(chan:int, chan_in=None, chan_out=None, depth:int=3, actv_mod=nn.SiLU):
  def make_linear(idx):
    dim_in  = chan_in  if idx == 0         and chan_in  is not None else chan
    dim_out = chan_out if idx + 1 == depth and chan_out is not None else chan
    return nn.Linear(dim_in, dim_out)
  modules = [make_linear(0)]
  for i in range(1, depth):
    modules.append(actv_mod())
    modules.append(make_linear(i))
  return nn.Sequential(*modules)

class GatedMLP(nn.Module):
  def __init__(self, chan:int, actv_mod=nn.SiLU):
    super().__init__()
    self.lin_0 = nn.Linear(chan, chan)
    self.actv_1 = actv_mod()
    self.lin_gate_1 = nn.Linear(chan, chan)
    self.lin_1 = nn.Linear(chan, chan)
    self.actv_2 = actv_mod()
    self.lin_gate_2 = nn.Linear(chan, chan)
    self.lin_2 = nn.Linear(chan, chan)
  def forward(self, x, enc):
    return self.lin_2(self.lin_gate_2(enc)*self.actv_2(
      self.lin_1(self.lin_gate_1(enc)*self.actv_1(
        self.lin_0(x)
      ))
    ))


class LocalProds(nn.Module):
  def __init__(self, chan, rank):
    super().__init__()
    self.prod_o0 = ElementwiseTensorProd_o0(rank)
    self.prod_o1 = ElementwiseTensorProd_o1(rank)
    self.lin_l0 = nn.Linear(chan, rank)
    self.lin_r0 = nn.Linear(chan, rank)
    self.lin_l1 = nn.Linear(chan, rank, bias=False)
    self.lin_r1 = nn.Linear(chan, rank, bias=False)
    self.lin_l0_new = nn.Linear(chan, rank)
    self.lin_r0_new = nn.Linear(chan, rank)
    self.lin_o0 = nn.Linear(2*rank, chan)                   # 2 kinds of products make scalars
    self.lin_o1 = VecGatedLinear(3*rank, chan, chan, rank)  # 3 kinds of products make vectors
    self.mlp = GatedMLP(chan)
  def forward(self, z_0, z_1, enc):
    z_l0, z_r0 = self.lin_l0(z_0), self.lin_r0(z_0)
    z_l1, z_r1 = self.lin_l1(z_1), self.lin_r1(z_1)
    z_0 = z_0 + self.lin_o0(torch.tanh(self.prod_o0(
      z_l0, z_l1, z_r0, z_r1)))
    z_0 = self.mlp(z_0, enc)
    z_l0, z_r0 = self.lin_l0_new(z_0), self.lin_r0_new(z_0) # update to new values after we ran the MLP
    z_1 = self.lin_o1(vector_sigmoid(self.prod_o1(
      z_l0, z_l1, z_r0, z_r1)), enc)
    return z_0, z_1


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


class ResidueAtomEmbed(nn.Module):
  # Note: this implementation is only compatible with openmm sims where allow_H=True
  res_lens = { # residue lengths including hydrogens & not including terminal atoms
    "A": 10, "V": 16, "I": 19, "L": 19, "M": 17, "F": 20, "Y": 21, "W": 24, "S": 11,
    "T": 14, "N": 14, "Q": 17, "R": 24, "H": 17, "K": 22, "D": 12, "E": 15, "C": 11,
    "G": 7, "P": 14,
  }
  # count up atoms in resiudes and create index mapping
  letters = [letter for letter in res_lens]
  letters.sort() # make sure ordering is always the same
  idx_curr = 0
  index_mapping = {} # what index into the embeddings would the first atom (backbone Nitrogen) have?
  for letter in letters:
    index_mapping[letter] = idx_curr
    idx_curr += res_lens[letter]
  def __init__(self, dim):
    super().__init__()
    self.embeddings = nn.Parameter(torch.zeros(self.idx_curr, dim))
  def _get_index(self, res, i):
    return self.index_mapping[res] + i
  def forward(self, metadata:OpenMMMetadata):
    """ ans: (atoms, dim) """
    return self.embeddings[torch.cat([
        torch.arange(self.res_lens[letter], device=self.embeddings.device) + self.index_mapping[letter]
        for letter in metadata.seq
      ])]


class PosEmbed(nn.Module):
  """ Embeddings of node-wise relative positions. """
  def __init__(self, chan:int):
    super().__init__()
    self.lin_v = nn.Linear(1, chan, bias=False)
  def forward(self, pos_0, pos_1):
    """ pos_0, pos_1: (batch, nodes, 3)
        ans: (batch, nodes, 3, chan) """
    pos_0, pos_1 = pos_0[..., None], pos_1[..., None]
    dpos_v = 0.1*(pos_1 - pos_0)
    return self.lin_v(dpos_v)


class Messages(nn.Module):
  def __init__(self, r0:float, chan:int, rank:int):
    super().__init__()
    self.r0 = r0
    # submodules
    self.contract = MessageContract(rank, chan, chan)
    self.lin_enc = nn.Linear(8, chan)
    self.lin_src = nn.Linear(chan, chan)
    self.lin_dst = nn.Linear(chan, chan)
  def forward(self, graph, r_ij, z_0, z_1, emb):
    emb_i, emb_j = edges_read(graph, emb)
    emb_ij = self.lin_src(emb_i) + self.lin_dst(emb_j)
    dist_emb_ij = self.lin_enc(radial_encode_8(r_ij, self.r0)) + emb_ij # relu for continuity included in radial_encode_8
    r_ij = vector_sigmoid(r_ij[..., None]*(7./self.r0)).squeeze(-1) # soft-normalize radial separations
    z_0_j = edges_read_dst(graph, z_0)
    z_1_j = edges_read_dst(graph, z_1)
    ψ_0_ij, ψ_1_ij = self.contract(z_0_j, z_1_j, r_ij, dist_emb_ij)
    return edges_reduce_src(graph, ψ_0_ij), edges_reduce_src(graph, ψ_1_ij)


class MessagesNocut(nn.Module):
  """ Messages layer but without a cutoff radius (accept given graph with infinite range) """
  def __init__(self, chan:int, rank:int, edgelabel_dim:int):
    super().__init__()
    # submodules
    self.contract = MessageContract(rank, chan, chan)
    self.lin_label = nn.Linear(edgelabel_dim, chan)
    self.lin_src = nn.Linear(chan, chan)
    self.lin_dst = nn.Linear(chan, chan)
  def forward(self, graph, pos, z_0, z_1, emb, edgelabels):
    pos_i, pos_j = edges_read(graph, pos)
    r_ij = 0.1*(pos_j - pos_i)
    z_0_j = edges_read_dst(graph, z_0)
    z_1_j = edges_read_dst(graph, z_1)
    emb_i, emb_j = edges_read(graph, emb)
    emb_ij = self.lin_src(emb_i) + self.lin_dst(emb_j)
    ψ_0_ij, ψ_1_ij = self.contract(z_0_j, z_1_j, r_ij, emb_ij + self.lin_label(edgelabels))
    return edges_reduce_src(graph, ψ_0_ij), edges_reduce_src(graph, ψ_1_ij)


def tupsum(*tups):
  ans = list(tups[0])
  for tup in tups[1:]: # slice off 0 since we already did it
    for i, elem in enumerate(tup):
      ans[i] = ans[i] + elem
  return tuple(ans)



class Block(nn.Module):
  def __init__(self, r0:float, chan:int, rank:int, neighbours_max:int=64):
    super().__init__()
    self.neighbours_max = neighbours_max
    self.r0 = r0
    # submodules
    self.s_emb = TimeEmbedding(chan//2, chan)
    self.atm_emb = ResidueAtomEmbed(chan)
    self.actv_a_emb = nn.Linear(chan, chan)
    self.pos_emb = PosEmbed(chan)
    self.messages_0 = Messages(r0, chan, rank)
    self.messages_1 = Messages(r0, chan, rank)
    self.messages_aminos_0 = MessagesNocut(chan, rank, 3)
    self.messages_aminos_1 = MessagesNocut(chan, rank, 3)
    self.local_prods = LocalProds(chan, rank)
    self.lin_push_pos_1 = nn.Linear(chan, 1, bias=False)
  def forward(self, tup):
    """ all args are bundled into a tup so we can stack Block layers easily
      s: (batch)
      pos_0, pos_1: (batch, atoms, 3)
      z_0: (batch, atoms, chan)
      z_1: (batch, atoms, 3, chan)
    """
    s, pos_0_tup, pos_1_tup, z_tup, contexttup = tup
    z_0, z_1 = z_tup
    pos_0, graph_0, r_ij_0 = pos_0_tup
    pos_1, graph_1, r_ij_1 = pos_1_tup
    box, metadata, graph_aminos, graph_aminos_edgelabels = contexttup
    # redo graph setup if needed
    if graph_0 is None: # pos_0 was modified since graph was last computed
      graph_0, r_ij_0 = graph_setup(self.r0, box, pos_0, neighbours_max=self.neighbours_max)
      pos_0_tup = pos_0, graph_0, r_ij_0
    if graph_1 is None: # pos_1 was modified since graph was last computed
      graph_1, r_ij_1 = graph_setup(self.r0, box, pos_1, neighbours_max=self.neighbours_max)
      pos_1_tup = pos_1, graph_1, r_ij_1
    # get the embedding
    emb = self.s_emb(s)[:, None] + self.atm_emb(metadata) + torch.tanh(self.actv_a_emb(z_0))
    # add vector embedding of the difference between positions
    z_1 = z_1 + self.pos_emb(pos_0, pos_1)
    # apply amino-sequence-based messages
    z_0, z_1 = tupsum(
      (z_0, z_1),
      self.messages_aminos_0(graph_aminos, pos_0, z_0, z_1, emb, graph_aminos_edgelabels),
      self.messages_aminos_1(graph_aminos, pos_1, z_0, z_1, emb, graph_aminos_edgelabels)
    )
    # apply proximity-based messages
    z_0, z_1 = tupsum(
      (z_0, z_1),
      self.messages_0(graph_0, r_ij_0, z_0, z_1, emb),
      self.messages_1(graph_1, r_ij_1, z_0, z_1, emb)
    )
    # local processing
    z_0, z_1 = tupsum(
      (z_0, z_1),
      self.local_prods(z_0, z_1, emb)
    )
    # update pos_1
    pos_1 = pos_1 + self.lin_push_pos_1(z_1).squeeze(-1)
    pos_1_tup = pos_1, None, None # invalidate previous graph and r_ij
    return s, pos_0_tup, pos_1_tup, (z_0, z_1), contexttup


def one_way_fcbg(inds_src:torch.Tensor, inds_dst:torch.Tensor):
  """ makes a directed fully-connected bipartite graph
      expects inds_src and inds_dst to be in sorted order """
  len_src, = inds_src.shape
  len_dst, = inds_dst.shape
  return (
    inds_src[:, None].expand(len_src, len_dst),
    inds_dst[None, :].expand(len_src, len_dst)
  )

def two_way_fcg(inds:torch.Tensor):
  """ makes a fully-connected graph
      expects inds to be in sorted order """
  len_inds, = inds.shape
  nodiag = ~torch.eye(len_inds, device=inds.device, dtype=torch.bool).reshape(len_inds*len_inds)
  return (
    inds[:, None].expand(len_inds, len_inds - 1),
    inds[None, :].expand(len_inds, len_inds).reshape(len_inds*len_inds)[nodiag].reshape(len_inds, len_inds - 1),
  )


def make_base_amino_graph(metadata:OpenMMMetadata, device):
  n_atoms, = metadata.atomic_nums.shape
  # find index ranges for atoms in each residue
  inds = []
  for res_start, res_end in zip(metadata.residue_indices[:-1], metadata.residue_indices[1:]):
    inds.append(torch.arange(res_start, res_end, device=device))
  else: # missed the last residue, do it at the end
    inds.append(torch.arange(metadata.residue_indices[-1], n_atoms, device=device))
  # create the graph
  flat_edges_src, flat_edges_dst = [], []
  flat_edge_labels = []
  for inds_prev, inds_curr, inds_next in zip([None] + inds[:-1], inds, inds[1:] + [None]):
    edges_src, edges_dst = [], []
    edge_labels = []
    if inds_prev is not None: # edges curr -> prev
      bwd_src, bwd_dst = one_way_fcbg(inds_curr, inds_prev)
      edges_src.append(bwd_src)
      edges_dst.append(bwd_dst)
      edge_labels.append(torch.full_like(bwd_src, 2, dtype=torch.long))
    if inds_curr is not None: # edges curr -> curr
      fcg_src, fcg_dst = two_way_fcg(inds_curr)
      edges_src.append(fcg_src)
      edges_dst.append(fcg_dst)
      edge_labels.append(torch.full_like(fcg_src, 0, dtype=torch.long))
    if inds_next is not None: # edges curr -> next
      fwd_src, fwd_dst = one_way_fcbg(inds_curr, inds_next)
      edges_src.append(fwd_src)
      edges_dst.append(fwd_dst)
      edge_labels.append(torch.full_like(fwd_src, 1, dtype=torch.long))
    flat_edges_src.append(torch.cat(edges_src, dim=-1).flatten())
    flat_edges_dst.append(torch.cat(edges_dst, dim=-1).flatten())
    flat_edge_labels.append(torch.cat(edge_labels, dim=-1).flatten())
  graph = Graph(torch.cat(flat_edges_src), torch.cat(flat_edges_dst), None, 1, n_atoms)
  edge_labels = torch.cat(flat_edge_labels)
  return graph, edge_labels


def get_amino_graph(metadata:OpenMMMetadata, device, batch:int):
  graph, edge_labels = make_base_amino_graph(metadata, device)
  graph = expand_graph(graph, batch)
  edge_labels = edge_labels.repeat(batch)
  edge_labels = F.one_hot(edge_labels, 3).to(torch.float32)
  return graph, edge_labels



class Denoiser(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.chan = config["chan"]
    # submodules
    self.blocks = nn.Sequential(*[
      Block(config["r_cut"], self.chan, config["rank"], config["neighbours_max"])
      for i in range(config["depth"])
    ])
    self.readout_v = nn.Linear(self.chan, 1, bias=False)
    self.amino_graph_cache = {}
  def _amino_graph(self, metadata:OpenMMMetadata, device, batch:int):
    if (metadata.seq, device, batch) not in self.amino_graph_cache:
      self.amino_graph_cache[metadata.seq, device, batch] = get_amino_graph(metadata, device, batch)
    return self.amino_graph_cache[metadata.seq, device, batch]
  def make_block_tup(self, s, pos_0, pos_1, box, metadata):
    device = s.device
    batch,          atoms,          must_be[3] = pos_0.shape
    must_be[batch], must_be[atoms], must_be[3] = pos_1.shape
    pos_0, pos_1 = pos_0.contiguous(), pos_1.contiguous()
    contexttup = box, metadata, *self._amino_graph(metadata, device, batch)
    z_0 = torch.zeros(batch, atoms, self.chan, device=device)
    z_1 = torch.zeros(batch, atoms, 3, self.chan, device=device)
    return s, (pos_0, None, None), (pos_1, None, None), (z_0, z_1), contexttup
  def forward(self, s, pos_0, pos_1, box, state:ModelState):
    tup = self.blocks(self.make_block_tup(s, pos_0, pos_1, box, state))
    return self.readout_v(tup[3][1]).squeeze(-1), tup[2][0]


def nodecay_cosine_schedule(t, sigma_max):
  return torch.cos(0.5*torch.pi*t)/torch.sqrt(sigma_max**-2 + torch.sin(0.5*torch.pi*t)**2)


class DiffusionDenoiser:
  is_gan = True
  def __init__(self, config):
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
    return loss
  def sigma_t(self, t):
    return nodecay_cosine_schedule(t, self.sigma_max)
  def _get_epsilon_pred(self, sigma, x_1_noised, nn_output):
    """ given a noise value sigma, noised sample, and the nn output, return the combined predicted epsilon """
    epsilon_pred, x_1_pred = nn_output
    zeta = self.config["zeta"]
    # for combination rule, see https://arxiv.org/pdf/2202.00512 appendix D
    return (zeta*epsilon_pred + (x_1_noised - x_1_pred))/(zeta + sigma)
  def _diffuser_step(self, x, metadata):
    """ x: (L, batch, poly_len, 3) """
    L, batch, atoms, must_be[3] = x.shape
    x_0 = x[:-1].reshape((L - 1)*batch, atoms, 3)
    x_1 = x[1:].reshape((L - 1)*batch, atoms, 3)
    t = torch.rand((L - 1)*batch, device=x.device)
    epsilon = torch.randn_like(x_0)
    sigma = self.sigma_t(t)[:, None, None]
    x_1_noised = x_1 + sigma*epsilon
    # call the network and compute predicted epsilon value
    epsilon_pred = self._get_epsilon_pred(sigma, x_1_noised,
      self.dn(t, x_0, x_1_noised, self.box, metadata))
    x_1_pred = x_1_noised - sigma*epsilon_pred
    loss = (((x_1_pred - x_1)/(0.4 + sigma))**2).mean()
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return loss.item()
  def generate(self, x_0, metadata, steps=20):
    *leading_dims, atoms, must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, atoms, 3)
    ans = x_0 + self.sigma_t(torch.zeros(1, device=x_0.device))[:, None, None]*torch.randn(batch, atoms, 3, device=x_0.device)
    t_list = np.linspace(0., 1., steps + 1)
    for i in range(steps):
      t = torch.tensor([t_list[i]], device=x_0.device, dtype=torch.float32)
      tdec= torch.tensor([t_list[i + 1]], device=x_0.device, dtype=torch.float32)
      sigma_t = self.sigma_t(t)[:, None, None]
      sigma_tdec = self.sigma_t(tdec)[:, None, None]
      dsigma = torch.sqrt(sigma_t**2 - sigma_tdec**2)
      epsilon_pred = self._get_epsilon_pred(sigma_t, ans,
        self.dn(t, x_0, ans, self.box, metadata))
      ans -= (dsigma**2/sigma_t)*epsilon_pred
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



if __name__ == "__main__":
  print("Testing symmetries of vector_sigmoid...")
  from managan.symmetries import check_symm, RotSymm
  def compute_gradient_fn(v, dv):
    y = vector_sigmoid(v)
    y.backward(dv)
    return y, v.grad
  v = torch.randn(10, 100, 3, 32, device="cuda", requires_grad=True)
  dv = torch.randn_like(v)
  check_symm(RotSymm(), compute_gradient_fn, [v, dv], ["1c", "1c"], ["1c", "1c"])
