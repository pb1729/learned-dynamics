import torch

from .utils import avg_relative_diff


class Symmetry:
  """ base symmetry class corresponding to the trivial symmetry group """
  def tens(self, inds:int, x:torch.Tensor, pos=-1):
    """ x: (..., (3,)^inds, ...)
        apply symmetry transform to tensor with indices, where indices are inserted into the
        rest of the dims at position pos """
    return x
  def pos(self, p_v):
    """ (..., nodes, 3)
        apply symmetry to a position vector """
    return p_v
  def _apply(self, tens, tens_type):
    if tens_type == "": # act trivially on non-tensor objects
      return tens
    assert isinstance(tens_type, str), "tensor type descriptors must be a string"
    assert isinstance(tens, torch.Tensor), "non-tensors must have symmetry type \"\""
    if tens_type == "p":
      return self.pos(tens)
    if tens_type[-1:] == "c":
      inds = int(tens_type[:-1])
      return self.tens(inds, tens, pos=-2)
    else:
      inds = int(tens_type)
      return self.tens(inds, tens)
  def apply(self, tens_list, tens_type_list):
    ans = []
    for tens, tens_type in zip(tens_list, tens_type_list):
      transformed = self._apply(tens, tens_type)
      if isinstance(tens, torch.Tensor) and tens.requires_grad: # maintain leaf tensors as leaf tensors
        transformed = transformed.detach().clone().requires_grad_()
      ans.append(transformed)
    return ans

def tensor_matrix_transform(inds:int, pos:int, x, R):
  """ transform all indices of the matrix by matrix R
      x:(..., (3,)^inds)
      R: (3, 3) """
  assert pos < 0, "positive pos not yet supported"
  assert pos > ord("a") - ord("i"), "too big a pos offset"
  cap_indices = "".join([chr(ord("I") + n) for n in range(inds)])
  lower_indices = "".join([chr(ord("i") + n) for n in range(inds)])
  chan_indices = "".join([chr(ord("a") + n) for n in range(-(pos + 1))])
  pairs = "".join([", " + cap + lower for cap, lower in zip(cap_indices, lower_indices)])
  einsum_str = f"...{cap_indices}{chan_indices}{pairs} -> ...{lower_indices}{chan_indices}"
  return torch.einsum(einsum_str, x, *([R]*inds))

class RotSymm(Symmetry):
  """ Check that rotating input by a rotation matrix is the same as rotating the output by that matrix. """
  random_rot = torch.tensor([
    [-0.2590, -0.5228,  0.8121],
    [ 0.6468, -0.7184, -0.2562],
    [ 0.7174,  0.4589,  0.5242]], device="cuda")
  def tens(self, inds:int, x:torch.Tensor, pos=-1):
    return tensor_matrix_transform(inds, pos, x, self.random_rot)
  def pos(self, p_v):
    return p_v @ self.random_rot

class TransSymm(Symmetry):
  """ Check symmetry under uniform translations of the whole system. """
  trans = torch.tensor([10.5, 26., -17.], device="cuda")
  def pos(self, p_v):
    return p_v + self.trans


def check_symm(symm, fn, args, arg_symms, output_symms):
  y = fn(*args)
  s_y = symm.apply(y, output_symms)
  args_s = symm.apply(args, arg_symms)
  y_s = fn(*args_s)
  print(symm.__class__.__name__)
  for tens_y_s, tens_s_y in zip(y_s, s_y):
    print(avg_relative_diff(tens_s_y, tens_y_s))
