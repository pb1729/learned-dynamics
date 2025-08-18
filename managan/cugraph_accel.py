import torch
from torch import nn


class IdModule(nn.Module):
  """ Seems useless, but provides a meat-shield to protect module from mutation by make_graphed_callables. """
  def __init__(self, mod):
    super().__init__()
    self.mod = mod
  def forward(self, *args):
    return self.mod(*args)


def cugraph_wrap_module(mod):
  """ Wraps a module with torch's make_graphed_callables for acceleration.
      We keep some size flexibility by storing a buffer of arg shapes. """
  cache = {}
  def get_sig(args):
    return tuple([
      (tuple(arg.shape), tuple(arg.stride()), arg.dtype, arg.requires_grad)
      for arg in args
    ])
  def get_sample_args(args):
    return tuple([
      arg.detach().clone().requires_grad_(arg.requires_grad)
      for arg in args
    ])
  def callmod(*args):
    sig = get_sig(args)
    if sig not in cache:
      cache[sig] = torch.cuda.make_graphed_callables(IdModule(mod), get_sample_args(args))
      if len(cache) > 1:
        print(f"WARNING: Captured additional signature {sig} for module with id {id(mod)}. "
          "This occupies some memory going forward and was probably quite slow to generate.")
    return cache[sig](*args)
  return callmod
