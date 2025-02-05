import torch
import torch.nn as nn

from .utils import must_be
from .seq2pdbchain.amino_data import letter_code, structures
from .sim_utils import OpenMMMetadata
from .layers_common import VecLinear


def get_residue_len(letter:str) -> int:
  struct = structures[letter_code[letter]][0]
  return len(struct.atoms) - 1 # subtract 1 to prevent N_next from being included


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
      letter: SingleResidueEncode(get_residue_len(letter), vdim, nlin=nlin)
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
      letter: SingleResidueDecode(get_residue_len(letter), vdim, nlin=nlin)
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
        VecLinear(vdim, get_residue_len(letter)))
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
