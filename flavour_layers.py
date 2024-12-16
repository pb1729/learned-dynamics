import torch
import torch.nn as nn

from utils import must_be
from openmm_sims import structures, letter_code, OpenMMMetadata
from layers_common import VecLinear


def get_residue_len(letter:str) -> int:
  struct = structures[letter_code[letter]][0]
  return len(struct.atoms) - 1 # subtract 1 to prevent N_next from being included


class SingleResidueEncode(nn.Module):
  def __init__(self, natom:int, vdim:int):
    super().__init__()
    self.natom = natom
    self.lin = VecLinear(natom, vdim)
  def forward(self, x:torch.Tensor):
    """ x:(batch, natom, 3)
        ans: (batch, vdim, 3) """
    root = x[:, 1] # Root is taken to be position of alpha carbon, which is always at index 1
    delta_x = x - root[:, None]
    return self.lin(delta_x)

class ResiduesEncode(nn.Module):
  def __init__(self, vdim:int):
    super().__init__()
    self.res_enc = nn.ModuleDict({
      letter: SingleResidueEncode(get_residue_len(letter), vdim)
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
  def __init__(self, natom:int, vdim:int):
    super().__init__()
    self.natom = natom
    self.lin = VecLinear(vdim, natom)
  def init_to_zeros(self):
    with torch.no_grad():
      self.lin.W.zero_()
  def forward(self, pos_ca:torch.Tensor, x_v:torch.Tensor):
    """ pos_ca: (batch, 3)
        x_v: (batch, vdim, 3)
        ans: (batch, natom, 3) """
    return pos_ca[:, None] + self.lin(x_v)

class ResiduesDecode(nn.Module):
  def __init__(self, vdim:int):
    super().__init__()
    self.res_dec = nn.ModuleDict({
      letter: SingleResidueDecode(get_residue_len(letter), vdim)
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
