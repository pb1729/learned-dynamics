from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .utils import must_be


class ErrorTrackingScheduler:
  def __init__(self, lr_fn):
    """ lr_fn: (step:int) -> float -- function that maps step to lr
        * Usage is to construct scheduler before optimizer and initialize optimizer lr to scheduler.lr() """
    self._lr_fn = lr_fn
    self._lr = self._lr_fn(0)
  def step(self, step_count:int, optim:torch.optim.Optimizer):
    lr_new = self._lr_fn(step_count)
    if abs(lr_new - self._lr)/(lr_new + self._lr) > 0.01:
      for group in optim.param_groups: # update the optimizer
        group["lr"] *= lr_new/self._lr # relative update
      self._lr = lr_new
  def lr(self):
    return self._lr

def get_lr_fn_warmup_and_decay(base_lr:float, start_fac:float, warmup_steps:int, gamma:float):
  """ base_lr: float -- the base learning rate of this scheduler
      start_fac: float -- in [0, 1], initial learning rate is start_fac*lr
      warmup_steps: int -- number of steps for linear warmup from lr to start_fac
      gamma: float -- relative decay per step, decay across n steps is roughly (1 - gamma)**n """
  def lr_fn(step:int):
    t = step - warmup_steps
    return base_lr*min(
      1. + (1. - start_fac)*t/warmup_steps,
      np.exp(gamma*t)
    )
  return lr_fn


# GAN stuff:
def taxicab_dist(x1, x2, epsilon=0.01):
  """ x1, x2: (batch, poly_len, 3)
      ans: (batch) """
  return torch.sqrt(((x1 - x2)**2).sum(-1) + epsilon).mean(-1)
def l2_dist(x1, x2, epsilon=0.01):
  """ x1, x2: (batch, poly_len, 3)
      ans: (batch) """
  return torch.sqrt(((x1 - x2)**2).sum(-1).mean(-1))
def endpoint_penalty(x1, x2, y1, y2, lambda_wass:Optional[float], lambda_l2:Optional[float], epsilon:float=0.01):
  """ functional form of endpoint penalty
      x1, x2: (batch, poly_len, 3)
      y1, y2: (batch)
      ans: () """
  ans = 0.
  if lambda_wass is not None: # wasserstein penalty
    dist_taxi = taxicab_dist(x1, x2, epsilon=epsilon) # (batch)
    penalty_wass = F.relu(torch.abs(y1 - y2)/dist_taxi - 1.) # (batch)
    ans = ans + lambda_wass*penalty_wass.mean()
  if lambda_l2 is not None: # zero-centered l2 penalty
    dist_l2 = l2_dist(x1, x2, epsilon=epsilon) # (batch)
    penalty_l2 = 0.5*((y1 - y2)/dist_l2)**2 # (batch)
    ans = ans + lambda_l2*penalty_l2.mean()
  return ans
def get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g, box, metadata, lambda_wass:Optional[float], lambda_l2:Optional[float]):
  """ full computation of endpoint penalty on interpolated data
      x_0: (batch, poly_len, 3)
      x_r, x_g: (batch, poly_len, 3)
      y_r, y_g: (batch)
      ans: () """
  batch, nodes, must_be[3] = x_0.shape
  assert x_r.shape == x_0.shape == x_g.shape
  must_be[batch], = y_r.shape
  must_be[batch], = y_g.shape
  mix_factors = torch.rand(batch, 1, 1, device=x_0.device)
  x_m = mix_factors*x_g + (1 - mix_factors)*x_r
  y_m = disc(x_0, x_m, box, metadata)
  return (endpoint_penalty(x_r, x_g, y_r, y_g, lambda_wass, lambda_l2)
        + endpoint_penalty(x_r, x_m, y_r, y_m, lambda_wass, lambda_l2)
        + endpoint_penalty(x_g, x_m, y_g, y_m, lambda_wass, lambda_l2))

class NormalizeGradient(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    return x
  @staticmethod
  def backward(ctx, dx:torch.Tensor):
    mag = (dx**2).mean(tuple(range(1, len(dx.shape))), keepdim=True)
    return dx/torch.sqrt(1e-10 + mag)
norm_grad = NormalizeGradient.apply
