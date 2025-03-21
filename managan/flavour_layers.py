import numpy as np
import torch
import torch.nn as nn

from .utils import must_be
from .seq2pdbchain.amino_data import letter_code, structures
from .seq2pdbchain.amino_bonds import bond_graphs
from .sim_utils import OpenMMMetadata
from .layers_common import VecLinear
from .tensor_products import TensLinear
from .graph_layers import Graph


def get_residue_len(letter:str) -> int:
  struct = structures[letter_code[letter]][0]
  return len(struct.atoms) - 1 # subtract 1 to prevent N_next from being included

RES_LEN = {letter: get_residue_len(letter) for letter in letter_code}


class SingleResidueEncode(nn.Module):
  def __init__(self, natom:int, vdim:int, nlin:int=1):
    super().__init__()
    self.natom = natom
    self.layers = nn.Sequential(*(
      [VecLinear(natom, vdim)] +
      [
        VecLinear(vdim, vdim)
        for i in range(nlin - 1)
      ]))
  def forward(self, x:torch.Tensor):
    """ x:(batch, natom, 3)
        ans: (batch, vdim, 3) """
    root = x[:, 1] # Root is taken to be position of alpha carbon, which is always at index 1
    delta_x = x - root[:, None]
    return self.layers(delta_x)

class ResiduesEncode(nn.Module):
  def __init__(self, vdim:int, nlin:int=1):
    super().__init__()
    self.res_enc = nn.ModuleDict({
      letter: SingleResidueEncode(RES_LEN[letter], vdim, nlin=nlin)
      for letter in letter_code
    })
  def forward(self, x:torch.Tensor, metadata:OpenMMMetadata):
    """ x: (batch, atoms, 3)
        ans: (batch, residues, 3), (batch, residues, vdim, 3) """
    def encode(res_enc, i):
      res_idx = metadata.residue_indices[i]
      return res_enc(x[:, res_idx:res_idx+res_enc.natom])
    x_v = torch.stack([
      encode(self.res_enc[letter], i)
      for i, letter in enumerate(metadata.seq)
    ], dim=1) # dim=1 because batch
    pos_ca = x[:, metadata.residue_indices + 1] # alpha carbon positions
    return pos_ca, x_v


class SingleResidueDecode(nn.Module):
  def __init__(self, natom:int, vdim:int, nlin:int=1):
    super().__init__()
    self.natom = natom
    self.layers = nn.Sequential(*(
      [
        VecLinear(vdim, vdim)
        for i in range(nlin - 1)
      ] +
      [VecLinear(vdim, natom)]))
  def init_to_zeros(self):
    with torch.no_grad():
      self.layers[-1].W.zero_()
  def forward(self, pos_ca:torch.Tensor, x_v:torch.Tensor):
    """ pos_ca: (batch, 3)
        x_v: (batch, vdim, 3)
        ans: (batch, natom, 3) """
    return pos_ca[:, None] + self.layers(x_v)

class ResiduesDecode(nn.Module):
  def __init__(self, vdim:int, nlin:int=1):
    super().__init__()
    self.res_dec = nn.ModuleDict({
      letter: SingleResidueDecode(RES_LEN[letter], vdim, nlin=nlin)
      for letter in letter_code
    })
  def init_to_zeros(self):
    for res in self.res_dec:
      self.res_dec[res].init_to_zeros()
  def forward(self, pos_ca:torch.Tensor, x_v:torch.Tensor, metadata:OpenMMMetadata):
    """ pos_ca: (batch, residues, 3)
        x_v: (batch, residues, vdim, 3)
        ans: (batch, atoms, 3) """
    return torch.cat([
      self.res_dec[letter](pos_ca[:, i], x_v[:, i])
      for i, letter in enumerate(metadata.seq)
    ], dim=1) # dim=1 because batch


class ResidueEmbed(nn.Module):
  def __init__(self, adim):
    super().__init__()
    self.embeddings = nn.ParameterDict({
      letter: nn.Parameter(torch.empty(adim))
      for letter in letter_code
    })
  def self_init(self):
    with torch.no_grad():
      for letter in self.embeddings:
        self.embeddings[letter].zero_()
  def forward(self, metadata:OpenMMMetadata):
    """ ans: (1, nodes, adim) """
    return torch.stack([self.embeddings[res] for res in metadata.seq], dim=0)[None]


class ResiduesDecodeVec(nn.Module):
  """ decode to a translation-invariant vector for each non-hydrogen atom in the residue """
  def __init__(self, vdim:int, nlin:int=1):
    super().__init__()
    self.res_dec = nn.ModuleDict({
      letter: nn.Sequential(
        *[VecLinear(vdim, vdim) for i in range(nlin - 1)],
        VecLinear(vdim, RES_LEN[letter]))
      for letter in letter_code
    })
  def init_to_zeros(self):
    with torch.no_grad():
      for res in self.res_dec:
        self.res_dec[res][-1].W.zero_()
  def forward(self, x_v, metadata:OpenMMMetadata):
    return torch.cat([
      self.res_dec[letter](x_v[:, i])
      for i, letter in enumerate(metadata.seq)
    ], dim=1) # dim=1 because batch


# Code in the next section works in the domain of individual atoms!

class ResidueAtomEmbed(nn.Module):
  # count up atoms in resiudes and create index mapping
  letters = [letter for letter in letter_code]
  letters.sort() # make sure ordering is always the same
  idx_curr = 0
  index_mapping = {} # what index into the embeddings would the first atom (backbone Nitrogen) have?
  reslen_mapping = {} # size of residue?
  for letter in letters:
    reslen = RES_LEN[letter]
    index_mapping[letter] = idx_curr
    reslen_mapping[letter] = reslen
    idx_curr += reslen
  def __init__(self, dim):
    super().__init__()
    self.embeddings = nn.Parameter(torch.zeros(self.idx_curr, dim))
  def _get_index(self, res, i):
    return self.index_mapping[res] + i
  def forward(self, metadata:OpenMMMetadata):
    """ ans: (atoms, dim) """
    return self.embeddings[torch.cat([
        torch.arange(self.reslen_mapping[letter], device=self.embeddings.device) + self.index_mapping[letter]
        for letter in metadata.seq
      ])]


def get_bond_src_dst(metadata:OpenMMMetadata):
  """ get src and dst indices for all the bonds in a protein """
  srclist, dstlist = [], []
  # add all stored bonds
  for i, letter in enumerate(metadata.seq):
    src, dst = bond_graphs[letter]
    srclist.append(src + metadata.residue_indices[i])
    dstlist.append(dst + metadata.residue_indices[i])
  # add bonds between adjacent amino acids. relevant atoms are N (index 0) and C (index 2)
  idx_N = metadata.residue_indices[1:]
  idx_C = metadata.residue_indices[:-1] + 2
  srclist.extend([idx_N, idx_C])
  dstlist.extend([idx_C, idx_N])
  # consolidate the arrays
  src = torch.tensor(np.concatenate(srclist))
  dst = torch.tensor(np.concatenate(dstlist))
  return src, dst

def get_bond_graph(batch:int, metadata:OpenMMMetadata, device) -> Graph:
  src, dst = get_bond_src_dst(metadata)
  src, dst = src.to(device), dst.to(device)
  # Graph expects src to have indices in increasing order. So we should sort them now.
  sort_idx = torch.argsort(src)
  src = src[sort_idx]
  dst = dst[sort_idx]
  # We now have what we need to make a graph with batch size 1.
  # But to honour the batch parameter, we need to make copies.
  atoms, = metadata.atomic_nums.shape
  src = src[None].expand(batch, -1)
  dst = dst[None].expand(batch, -1)
  src = src + atoms*torch.arange(batch, device=device)[:, None]
  dst = dst + atoms*torch.arange(batch, device=device)[:, None]
  src = src.reshape(-1)
  dst = dst.reshape(-1)
  return Graph(src, dst, None, batch, atoms)


# Code in the next section moves data between the domains of atoms and amino acids!

class LinAtomToAmino(nn.Module):
  def __init__(self, inds, atom_dim, amino_dim):
    super().__init__()
    self.atom_dim = atom_dim
    self.maps = nn.ModuleDict({
      letter: TensLinear(inds, RES_LEN[letter]*atom_dim, amino_dim)
      for letter in letter_code
    })
  def forward(self, x, metadata:OpenMMMetadata):
    """ x: (batch, atoms, atom_dim, (3,)^inds)
        ans: (batch, nodes, amino_dim, (3,)^inds) """
    batch, atoms, must_be[self.atom_dim], *ind_dims = x.shape
    return torch.stack([
      self.maps[letter](
        x[:, metadata.residue_indices[i]:(metadata.residue_indices[i] + RES_LEN[letter])]
        .reshape(batch, RES_LEN[letter]*self.atom_dim, *ind_dims))
      for i, letter in enumerate(metadata.seq)
    ], dim=1)

class LinAminoToAtom(nn.Module):
  def __init__(self, inds, amino_dim, atom_dim):
    super().__init__()
    self.amino_dim = amino_dim
    self.atom_dim = atom_dim
    self.maps = nn.ModuleDict({
      letter: TensLinear(inds, amino_dim, RES_LEN[letter]*atom_dim)
      for letter in letter_code
    })
  def forward(self, x, metadata:OpenMMMetadata):
    """ x: (batch, nodes, amino_dim, (3,)^inds)
        ans: (batch, atoms, atom_dim, (3,)^inds) """
    batch, must_be[len(metadata.seq)], must_be[self.amino_dim], *ind_dims = x.shape
    return torch.cat([
      self.maps[letter](x[:, i]).reshape(batch, RES_LEN[letter], self.atom_dim, *ind_dims)
      for i, letter in enumerate(metadata.seq)
    ], dim=1)

class PosMix(nn.Module):
  """ Converts data about where the atoms and aminos are located relative to each other in
      space to x_v activations. """
  def __init__(self, dim_v_atm, dim_v_amn):
    super().__init__()
    self.dim_v_atm = dim_v_atm
    self.lin_amn = nn.ModuleDict({
      letter: TensLinear(1, RES_LEN[letter], dim_v_amn)
      for letter in letter_code
    })
    self.lin_atm = nn.ModuleDict({
      letter: TensLinear(1, RES_LEN[letter], RES_LEN[letter]*dim_v_atm)
      for letter in letter_code
    })
  def forward(self, pos_atm, pos_amn, metadata:OpenMMMetadata):
    """ pos_atm: (batch, atoms, 3)
        pos_amn: (batch, nodes, 3)
        ans: x_v_atm, x_v_amn
        x_v_atm: (batch, atoms, dim_v_atm, 3)
        x_v_amn: (batch, nodes, dim_v_amn, 3) """
    batch, must_be[len(metadata.seq)], must_be[3] = pos_amn.shape
    must_be[batch], atoms, must_be[3] = pos_atm.shape
    diffs = [
      pos_atm[:, metadata.residue_indices[i]:(metadata.residue_indices[i] + RES_LEN[letter])] - pos_amn[:, i, None]
      for i, letter in enumerate(metadata.seq)] # [(batch, reslen_i, 3) i:nodes]
    x_v_amn = torch.stack([
      self.lin_amn[letter](diffs[i])
      for i, letter in enumerate(metadata.seq)
    ], dim=1)
    x_v_atm = torch.cat([
      self.lin_atm[letter](diffs[i]).reshape(batch, RES_LEN[letter], self.dim_v_atm, 3)
      for i, letter in enumerate(metadata.seq)
    ], dim=1)
    return x_v_atm, x_v_amn
