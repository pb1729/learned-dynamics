import torch
import torch.nn as nn

import cuequivariance_torch as cueq_torch

from .utils import must_be
from .tp_construction_cueq import make_elementwise_tensor_product, make_spherical_harmonics_tensor_product
from .fast_tp_unif_1d_jit import FastSegmentedPolynomialFromUniform1dJit


def _flatten_irrep(irrep_l:int, x):
  """ x: (..., irrep_dim, chan)
      ans: (..., irrep_dim*chan) """
  irrep_dim = 2*irrep_l + 1
  *rest, must_be[irrep_dim], chan = x.shape
  return x.reshape(*rest, irrep_dim*chan)

def _transpose_irrep_muls_to_channels(rank:int, num_irreps:int, irrep_l:int, x):
  """ Converts tensor x with irreps `{rank}x{l}`*num_irreps in ir_mul order to `{rank*num_irreps}x{x}` in ir_mul order. """
  irrep_dim = 2*irrep_l + 1
  *rest, must_be[rank*num_irreps*irrep_dim] = x.shape
  x = x.reshape(*rest, num_irreps, irrep_dim, rank)
  x = x.transpose(-2, -3)
  x = x.reshape(*rest, irrep_dim, num_irreps*rank)
  return x

class ElementwiseTensorProd_o0(nn.Module):
  def __init__(self, rank):
    super().__init__()
    poly_o0, irreps_o0 = make_elementwise_tensor_product(rank, [(0, 0, 0), (1, 1, 0)])
    self.tens_prod_o0 = FastSegmentedPolynomialFromUniform1dJit(poly_o0, name="elementwise_o0")
  def forward(self, z_0_l, z_1_l, z_0_r, z_1_r):
    z_l = torch.cat([z_0_l, _flatten_irrep(1, z_1_l)], dim=-1)
    z_r = torch.cat([z_0_r, _flatten_irrep(1, z_1_r)], dim=-1)
    z_0_o, = self.tens_prod_o0([z_l, z_r]) # pattern-match because output is a list
    return z_0_o

class ElementwiseTensorProd_o1(nn.Module):
  def __init__(self, rank):
    super().__init__()
    poly_o1, irreps_o1 = make_elementwise_tensor_product(rank, [(0, 1, 1), (1, 0, 1), (1, 1, 1)])
    self.tens_prod_o1 = FastSegmentedPolynomialFromUniform1dJit(poly_o1, name="elementwise_o1")
    self.rank = rank
    self.num_o1_irreps = len(irreps_o1)
  def forward(self, z_0_l, z_1_l, z_0_r, z_1_r):
    z_l = torch.cat([z_0_l, _flatten_irrep(1, z_1_l)], dim=-1)
    z_r = torch.cat([z_0_r, _flatten_irrep(1, z_1_r)], dim=-1)
    z_1_o, = self.tens_prod_o1([z_l, z_r]) # pattern-match because output is a list
    return _transpose_irrep_muls_to_channels(self.rank, self.num_o1_irreps, 1, z_1_o)


class ElementwiseTensorProducts(nn.Module):
  def __init__(self, rank, chan):
    super().__init__()
    self.lin_0_l = nn.Linear(chan, rank)
    self.lin_0_r = nn.Linear(chan, rank)
    self.lin_1_l = nn.Linear(chan, rank, bias=False)
    self.lin_1_r = nn.Linear(chan, rank, bias=False)
    self.lin_0_o = nn.Linear(2*rank, chan)              # 2 kinds of products make scalars
    self.lin_1_o = nn.Linear(3*rank, chan, bias=False)  # 3 kinds of products make vectors
    self.etp_o0 = ElementwiseTensorProd_o0(rank)
    self.etp_o1 = ElementwiseTensorProd_o1(rank)
  def forward(self, z0, z1):
    z0_l = self.lin_0_l(z0)
    z0_r = self.lin_0_r(z0)
    z1_l = self.lin_1_l(z1)
    z1_r = self.lin_1_r(z1)
    z0_o = self.etp_o0(z0_l, z1_l, z0_r, z1_r)
    z1_o = self.etp_o1(z0_l, z1_l, z0_r, z1_r)
    ans = self.lin_0_o(z0_o), self.lin_1_o(z1_o)
    return ans


class MessageContract(nn.Module):
  def __init__(self, rank, chan, dist_emb_dim, actv_mod=nn.SiLU):
    super().__init__()
    poly_o0, irreps_o0 = make_elementwise_tensor_product(rank, [(0, 0, 0), (1, 1, 0)], r_irreps=[0, 1, 2])
    poly_o1, irreps_o1 = make_elementwise_tensor_product(rank, [(0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 2, 1)], r_irreps=[0, 1, 2])
    self.lin_0_actv = nn.Linear(chan, rank)
    self.lin_1_actv = nn.Linear(chan, rank, bias=False)
    self.dist_mlp = nn.Sequential(
      nn.Linear(dist_emb_dim, rank),
      actv_mod(),
      nn.Linear(rank, rank),
      actv_mod(),
      nn.Linear(rank, rank))
    self.lin_0_o = nn.Linear(irreps_o0.count("0"), chan)
    self.lin_1_o = nn.Linear(irreps_o1.count("1"), chan, bias=False)
    poly_sh = make_spherical_harmonics_tensor_product([0, 1, 2])
    self.sh = FastSegmentedPolynomialFromUniform1dJit(poly_sh, name="sh_012")
    self.tens_prod_o0 = FastSegmentedPolynomialFromUniform1dJit(poly_o0, name="messages_o0")
    self.tens_prod_o1 = FastSegmentedPolynomialFromUniform1dJit(poly_o1, name="messages_o1")
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
    sh_ij, = self.sh([dir_ij])
    sh_dist_emb_ij = sh_ij[..., :, None]*dist_emb_ij[..., None, :] # (irrep, rank) ordering
    z_j = torch.cat([z0_j, _flatten_irrep(1, z1_j)], dim=-1)
    z0_o, = self.tens_prod_o0([z_j, sh_dist_emb_ij.reshape(-1, (1+3+5)*self.rank)])
    z1_o, = self.tens_prod_o1([z_j, sh_dist_emb_ij.reshape(-1, (1+3+5)*self.rank)])
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

  print("BWD_PASS_CHECK")
  # Check backward pass for etp_layer
  z_0_orig = z_0.clone().detach().requires_grad_(True)
  z_1_orig = z_1.clone().detach().requires_grad_(True)

  # Forward pass
  out_0, out_1 = etp_layer(z_0_orig, z_1_orig)
  print(out_0.grad_fn.next_functions)

  # Create random gradients for outputs
  grad_out_0 = torch.randn_like(out_0)
  grad_out_1 = torch.randn_like(out_1)

  # Backward pass
  loss =  (out_0 * grad_out_0).sum() + (out_1 * grad_out_1).sum()
  loss.backward()
  print(z_0_orig.grad)

  # Store gradients
  z_0_grad = z_0_orig.grad.clone()
  z_1_grad = z_1_orig.grad.clone()

  # Numerical check
  epsilon = 1e-4
  z_0_perturb = torch.randn_like(z_0) * epsilon
  z_1_perturb = torch.randn_like(z_1) * epsilon

  # Predicted change in output based on gradients
  pred_change_0 = (z_0_perturb * z_0_grad).sum() + (z_1_perturb * z_1_grad).sum()

  # Actual change in output by computing perturbed forward pass
  z_0_new = z_0_orig.detach() + z_0_perturb
  z_1_new = z_1_orig.detach() + z_1_perturb
  out_0_new, out_1_new = etp_layer(z_0_new, z_1_new)
  actual_change_0 = ((out_0_new - out_0) * grad_out_0).sum() + ((out_1_new - out_1) * grad_out_1).sum()

  print(f"Backward pass check: predicted change = {pred_change_0.item():.6e}, actual change = {actual_change_0.item():.6e}")
  print(f"Relative error: {abs(pred_change_0 - actual_change_0) / (abs(actual_change_0) + 1e-10):.6e}")

  # Check symmetry properties of backward pass
  print("Symmetry Checking Backward Pass...")

  # Create symmetric gradient tensors for testing
  grad_out_0 = torch.randn_like(out_0)
  grad_out_1 = torch.randn_like(out_1)

  # For ElementwiseTensorProducts
  def etp_with_grad(z_0, z_1, grad_0, grad_1):
    z_0 = z_0.clone().detach().requires_grad_(True)
    z_1 = z_1.clone().detach().requires_grad_(True)
    out_0, out_1 = etp_layer(z_0, z_1)
    loss = (out_0 * grad_0).sum() + (out_1 * grad_1).sum()
    loss.backward()
    return z_0.grad, z_1.grad

  check_symm(RotSymm(), etp_with_grad,
    [z_0, z_1, grad_out_0, grad_out_1], ["0c", "1c", "0c", "1c"], ["0c", "1c"])

  # For MessageContract
  def msg_with_grad(z_0, z_1, dir_ij, dist_emb_ij, grad_0, grad_1):
    z_0 = z_0.clone().detach().requires_grad_(True)
    z_1 = z_1.clone().detach().requires_grad_(True)
    dir_ij = dir_ij.clone().detach().requires_grad_(True)
    dist_emb_ij = dist_emb_ij.clone().detach().requires_grad_(True)
    out_0, out_1 = msg_layer(z_0, z_1, dir_ij, dist_emb_ij)
    loss = (out_0 * grad_0).sum() + (out_1 * grad_1).sum()
    loss.backward()
    return z_0.grad, z_1.grad, dir_ij.grad, dist_emb_ij.grad

  check_symm(RotSymm(), msg_with_grad,
    [z_0, z_1, dir_ij, dist_emb_ij, grad_out_0, grad_out_1],
    ["0c", "1c", "1", "0c", "0c", "1c"],
    ["0c", "1c", "1", "0c"])

  print("Backward Pass Symmetry Check Done.")
