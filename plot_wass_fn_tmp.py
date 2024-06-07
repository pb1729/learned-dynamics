import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import must_be
from config import load
from sims import sims, get_dataset
from wass_utils import *


RES = 16


def disc(model, x0, x1):
  return model.disc(x0, x1, model.graph)

def gen(model, x0):
  poly_len = model.config.sim.poly_len
  space_dim = model.config.sim.space_dim
  return model.predict(model.config.cond(x0.reshape(-1, poly_len*space_dim))).reshape(-1, poly_len, space_dim)

def proj(x):
  """ non-linearly project coords to 2d
      x: (batch, 2, 3) -- assumed to have a Center of Mass (COM) mode set to 0
      return: tuple((batch), (batch)) """
  delta = x[:, 1] -  x[:, 0]
  r_perp = torch.sqrt((delta[:, 1:]**2).sum(1))
  return delta[:, 0], r_perp


def get_continuation_dataset_x_only(sim, contins, x_init, v_init=None):
  """ get a dataset of many possible continued trajectories from an initial state
      contins: int
      x_init: (batch, q_dim)
      v_init: (batch, q_dim)
      return: (batch, contins, q_dim) --> gives the x coordinate only! """
  if v_init is None:
    v_init = torch.zeros_like(x_init) # default initial velocity is 0
  # expand along another dim to enumerate continuations
  batch,          q_dim          = x_init.shape
  must_be[batch], must_be[q_dim] = v_init.shape
  xv_init = torch.cat([x_init, v_init], dim=1) # (batch, 2*q_dim)
  xv_init = xv_init[:, None].expand(batch, contins, 2*q_dim) # (batch, contins, 2*q_dim)
  xv_init = xv_init.clone() # will be written to, so clone
  xv_init = xv_init.reshape(batch*contins, 2*q_dim)
  # calculate the continuations:
  xv_fin = get_dataset(sim, batch*contins, 1,
    x_init=xv_init[:, :q_dim], v_init=xv_init[:, q_dim:])
    # (batch*contins, 1, 2*q_dim)
  xv_fin = xv_fin.reshape(batch, contins, 2*q_dim)
  return xv_fin[:, :, :q_dim]


def main(fpath):
  assert fpath is not None
  # load the model
  model = load(fpath)
  model.set_eval(True)
  initial_state = torch.tensor([[-0.5, 0., 0.], [0.5, 0., 0.]], device="cuda")[None]
  with torch.no_grad():
    gens = gen(model, initial_state.expand(80000, -1, -1))
    gens = gens - gens.mean(1, keepdim=True) # not invariant, but it only zeros out the COM mode, which is decoupled
    z_gens = disc(model, initial_state.expand(80000, -1, -1), gens).detach()
    reals = get_continuation_dataset_x_only(model.config.sim, 80000, initial_state.reshape(1, -1).to(torch.float64))
    reals = reals.reshape(-1, model.config.sim.poly_len, model.config.sim.space_dim)
    reals = reals - reals.mean(1, keepdim=True) # not invariant, but it only zeros out the COM mode, which is decoupled
  x_gens, y_gens = proj(gens)
  x_reals, y_reals = proj(reals)
  p_gen, xedges, yedges = np.histogram2d(x_gens.cpu(), y_gens.cpu(), bins=[RES, RES], range=[[-2., 2.], [0., 2.5]])
  p_gen /= p_gen.sum()
  p_real, *_ = np.histogram2d(x_reals.cpu(), y_reals.cpu(), bins=[RES, RES], range=[[-2., 2.], [0., 2.5]])
  p_real /= p_real.sum()
  X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
  X, Y = X.flatten(), Y.flatten()
  p_gen = p_gen.flatten()
  p_real = p_real.flatten()
  print("comparison of real to generated distribution...")
  show_hist(X, Y, p_real)
  show_hist(X, Y, p_gen)
  show_hist(X, Y, p_gen, p_real)
  print("compare the Wasserstein dual functions:")
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(x_gens[:300].cpu(), y_gens[:300].cpu(), z_gens[:300].cpu())
  distances = xy_to_distances(X, Y)
  z_ideal = compute_optimal_lipschitz(p_gen, p_real, distances)
  ax.scatter(X, Y, z_ideal)
  plt.show()


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])






