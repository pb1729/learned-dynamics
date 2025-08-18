import torch
import torch.nn as nn

import cuequivariance_torch as cueq_torch

from .utils import must_be
from .tp_construction_cueq import make_elementwise_tensor_product


def _flatten_vec(x):
  """ x: (..., 3, chan)
      ans: (..., 3*chan) """
  *rest, must_be[3], chan = x.shape
  return x.reshape(*rest, 3*chan)

def _transpose_irrep_muls_to_channels(rank, num_irreps, irrep_l, x):
  """ Converts tensor x with irreps `{rank}x{l}`*num_irreps in ir_mul order to `{rank*num_irreps}x{x}` in ir_mul order. """
  irrep_dim = 2*irrep_l + 1
  *rest, must_be[rank*num_irreps*irrep_dim] = x.shape
  x = x.reshape(*rest, num_irreps, irrep_dim, rank)
  x = x.transpose(-2, -3)
  x = x.reshape(*rest, irrep_dim, num_irreps*rank)
  return x

class ElementwiseTensorProducts(nn.Module):
  def __init__(self, rank, chan):
    super().__init__()
    poly_o0, irreps_o0 = make_elementwise_tensor_product(rank, [(0, 0, 0), (1, 1, 0)])
    poly_o1, irreps_o1 = make_elementwise_tensor_product(rank, [(0, 1, 1), (1, 0, 1), (1, 1, 1)])
    self.lin_0_l = nn.Linear(chan, rank)
    self.lin_0_r = nn.Linear(chan, rank)
    self.lin_1_l = nn.Linear(chan, rank, bias=False)
    self.lin_1_r = nn.Linear(chan, rank, bias=False)
    self.lin_0_o = nn.Linear(irreps_o0.count("0"), chan)
    self.lin_1_o = nn.Linear(irreps_o1.count("1"), chan, bias=False)
    self.tens_prod_o0 = cueq_torch.SegmentedPolynomial(poly_o0, name="elementwise_o0")
    self.tens_prod_o1 = cueq_torch.SegmentedPolynomial(poly_o1, name="elementwise_o1")
    self.rank = rank
    self.num_o1_irreps = len(irreps_o1)
  def forward(self, z0, z1):
    z0_l = self.lin_0_l(z0)
    z0_r = self.lin_0_r(z0)
    z1_l = self.lin_1_l(z1)
    z1_r = self.lin_1_r(z1)
    z_l = torch.cat([z0_l, _flatten_vec(z1_l)], -1)
    z_r = torch.cat([z0_r, _flatten_vec(z1_r)], -1)
    z0_o, = self.tens_prod_o0([z_l, z_r]) # pattern-match because output is a list
    z1_o, = self.tens_prod_o1([z_l, z_r]) # pattern-match because output is a list
    z1_o = _transpose_irrep_muls_to_channels(self.rank, self.num_o1_irreps, 1, z1_o)
    ans = self.lin_0_o(z0_o), self.lin_1_o(z1_o)
    return ans


class MessageContract(nn.Module):
  def __init__(self, rank, chan, dist_emb_dim):
    super().__init__()
    poly_o0, irreps_o0 = make_elementwise_tensor_product(rank, [(0, 0, 0), (1, 1, 0)], r_irreps=[0, 1, 2])
    poly_o1, irreps_o1 = make_elementwise_tensor_product(rank, [(0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 2, 1)], r_irreps=[0, 1, 2])
    self.lin_0_actv = nn.Linear(chan, rank)
    self.lin_1_actv = nn.Linear(chan, rank, bias=False)
    self.dist_mlp = nn.Sequential(
      nn.Linear(dist_emb_dim, rank),
      nn.LeakyReLU(0.1),
      nn.Linear(rank, rank),
      nn.LeakyReLU(0.1),
      nn.Linear(rank, rank))
    self.lin_0_o = nn.Linear(irreps_o0.count("0"), chan)
    self.lin_1_o = nn.Linear(irreps_o1.count("1"), chan, bias=False)
    self.sh = cueq_torch.SphericalHarmonics([0, 1, 2], normalize=False)
    self.tens_prod_o0 = cueq_torch.SegmentedPolynomial(poly_o0, name="messages_o0")
    self.tens_prod_o1 = cueq_torch.SegmentedPolynomial(poly_o1, name="messages_o1")
    self.rank = rank
    self.num_o1_irreps = len(irreps_o1)
  def forward(self, z0_j, z1_j, dir_ij, dist_emb_ij):
    """ z0_j: (edges, chan)
        z1_j: (edges, 3, chan)
        dir_ij: (edges, 3)
        dist_emb_ij: (edges, dist_emb_dim)
        ans: (edges, chan), (edges, 3, chan) """
    z0_j = self.lin_0_actv(z0_j)
    z1_j = self.lin_1_actv(z1_j)
    dist_emb_ij = self.dist_mlp(dist_emb_ij)
    sh_ij = self.sh(dir_ij)
    sh_dist_emb_ij = sh_ij[..., :, None]*dist_emb_ij[..., None, :] # (irrep, rank) ordering
    z_j = torch.cat([z0_j, _flatten_vec(z1_j)], dim=-1)
    z0_o, = self.tens_prod_o0([z_j, sh_dist_emb_ij])
    z1_o, = self.tens_prod_o1([z_j, sh_dist_emb_ij])
    z1_o = _transpose_irrep_muls_to_channels(self.rank, self.num_o1_irreps, 1, z1_o)
    ans = self.lin_0_o(z0_o), self.lin_1_o(z1_o)
    return ans


if __name__ == "__main__":
  from .symmetries import check_symm, RotSymm
  DEV = "cuda"
  batch = 2080
  chan = 128
  rank = 256
  etp_layer = ElementwiseTensorProducts(rank, chan).to(DEV)
  z_0 = torch.randn(batch, chan, device=DEV)
  z_1 = torch.randn(batch, 3, chan, device=DEV)
  print("Symmetry Checking ElementwiseTensorProducts...")
  check_symm(RotSymm(), etp_layer,
    [z_0, z_1], ["0c", "1c"], ["0c", "1c"])
  print("Done.")
  msg_layer = MessageContract(rank, chan, chan//2).to(DEV)
  dist_emb_ij = torch.randn(batch, chan//2, device=DEV)
  dir_ij = torch.randn(batch, 3, device=DEV)
  print("Symmetry Checking MessageContract...")
  check_symm(RotSymm(), msg_layer,
    [z_0, z_1, dir_ij, dist_emb_ij], ["0c", "1c", "1", "0c"], ["0c", "1c"])
  print("Done.")
