from itertools import chain

import torch
from torch import nn


DEFAULT = "default"

def param_group_to_condition(params):
  return torch.cat([param.detach().flatten() for param in params])[None, :]

def get_grouped_parameters(module, as_lists=False):
  """ Returns parameters of the module organized by group.
      This function allows you to define modules that produce
      parameters that can belong to a variety of different
      groups beside the "default" group. Parameters in ordinary
      modules still belong to "default", however.
      Output ordering is consistent, so long as you define your fns
      list_groups() and grouped_parameters() to be consistent, see:
      https://discuss.pytorch.org/t/how-does-module-children-know-how-to-sort-submodules/58473 """
  ans = {}
  def append_params(group, params):
    if group in ans:
      ans[group] = chain(ans[group], params)
    else:
      ans[group] = params
  if hasattr(module, "list_groups"):
    for group in module.list_groups():
      append_params(group, module.grouped_parameters(group))
  else:
    # this is a normal module, its direct parameters are "default"
    append_params(DEFAULT, module.parameters(recurse=False))
  # get grouped parameters for all children
  for child in module.children():
    child_ans = get_grouped_parameters(child) # recursive call to get parameters
    for group in child_ans:
      append_params(group, child_ans[group])
  if as_lists: return {group: list(ans[group]) for group in ans}
  return ans


class FastParamsLinear(nn.Module):
  """ This module provides a low-rank linear transform.
      (some of) the parameters associated with this transform can
      be updated with a larger learning rate, hence the name "fast".
      These are intended for use in the generator network of a GAN,
      so that the discriminator can condition on these fast
      parameters. ch_middle gives the rank. """
  def __init__(self, ch_in, ch_middle, ch_out):
    super().__init__()
    self.lin_in = nn.Linear(ch_in, ch_middle, bias=False)
    self.w_mid = nn.Parameter(torch.randn(ch_middle, ch_middle))
    self.lin_out = nn.Linear(ch_middle, ch_out, bias=False)
  def forward(self, x):
    return self.lin_out(self.lin_in(x) @ self.w_mid)
  def list_groups(self):
    return (DEFAULT, "fast")
  def grouped_parameters(self, group):
    if group == "fast":
      return (p for p in [self.w_mid])
    return chain(self.lin_in.parameters(), self.lin_out.parameters())

class FastParamsConv1d(nn.Module):
  """ This module provides a low-rank convolution transform.
      (some of) the parameters associated with this transform can
      be updated with a larger learning rate, hence the name "fast".
      These are intended for use in the generator network of a GAN,
      so that the discriminator can condition on these fast
      parameters. ch_middle gives the rank. """
  def __init__(self, ch_in, ch_middle, ch_out, kernsz):
    super().__init__()
    self.conv_in = nn.Conv1d(ch_in, ch_middle, kernsz, padding="same", bias=False)
    self.w_mid = nn.Parameter(torch.randn(ch_middle, ch_middle))
    self.conv_out = nn.Conv1d(ch_middle, ch_out, kernsz, padding="same", bias=False)
  def forward(self, x):
    return self.conv_out(torch.einsum("bcx, cd -> bdx", self.conv_in(x), self.w_mid))
  def list_groups(self):
    return (DEFAULT, "fast")
  def grouped_parameters(self, group):
    if group == "fast":
      return (p for p in [self.w_mid])
    return chain(self.conv_in.parameters(), self.conv_out.parameters())










