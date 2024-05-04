import torch
import numpy as np
import matplotlib.pyplot as plt

from config import load
from sims import sims, get_dataset
from polymer_util import rouse, squarish_factorize


ROUSE_BASIS = False
SHOW_REALSPACE_SAMPLES = 10



def get_continuation_dataset(N, contins, sim, iterations=1, x_only=True):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...")
  initial_states = get_dataset(sim, N, 1, t_eql=10)[:, 0]
  print("created.")
  # expand along another dim to enumerate continuations
  state_dim = initial_states.shape[-1] # this should include velocity, so we can't use config.state_dim, which may be x only
  xv_init = initial_states[:, None].expand(N, contins, state_dim).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = get_dataset(sim, N*contins, iterations,
    x_init=xv_init[:, :state_dim//2], v_init=xv_init[:, state_dim//2:])
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, iterations, state_dim)
  if x_only:
    return initial_states[:, :state_dim//2], xv_fin[:, :, :, :state_dim//2]
  return initial_states, xv_fin


def compare_predictions_x(x_init, x_actual, sim, show_histogram=True, flash_plot=False):
  x_init = x_init.cpu().numpy()
  x_actual = x_actual.cpu().numpy()
  if SHOW_REALSPACE_SAMPLES > 0:
    for i in range(SHOW_REALSPACE_SAMPLES):
      plt.plot(x_actual[i], marker=".", color="blue", alpha=0.7)
    plt.plot(x_init[0], marker=".", color="black")
    plt.ylim(-5., 5.)
    if flash_plot:
      plt.show(block=False)
      plt.pause(0.7)
      plt.close()
    else:
      plt.show()
  if not show_histogram: return
  fig = plt.figure(figsize=(20, 12))
  axes = []
  plt_w, plt_h = squarish_factorize(sim.poly_len)
  for n in range(sim.poly_len):
    if ROUSE_BASIS:
      w_x = rouse(n, sim.poly_len)[None]
      x_init_n = (w_x*x_init).sum(1)
      x_actual_n = (w_x*x_actual).sum(1)
    else:
      x_init_n = x_init[:, n] - x_init[:, (n+1)%sim.poly_len]
      x_actual_n = x_actual[:, n] - x_actual[:, (n+1)%sim.poly_len]
    axes.append(fig.add_subplot(plt_h, plt_w, n + 1))
    axes[-1].hist(x_actual_n, range=(-2., 2.), bins=100, alpha=0.7)
    axes[-1].scatter([x_init_n], [0], color="black") # show starting point...
    axes[-1].scatter([x_actual_n.mean()], [0], color="blue")
  plt.show()


def eval_sample_step(init_states, fin_statess, sim):
  batch, contins, length, _ = fin_statess.shape
  for i in range(batch):
    print("\nnext i.c: i=%d\n" % i)
    for t in range(length):
      init_state = init_states[i, None]
      fin_states = fin_statess[i, :, t]
      compare_predictions_x(init_state, fin_states, sim)


def main(sim_nm):
  # get comparison data
  sim = sims[sim_nm]
  init_states, fin_states = get_continuation_dataset(10, 10000, sim)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  # compare!
  eval_sample_step(init_states, fin_states, sim)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])







