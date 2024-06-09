import torch
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from plotting_common import Plotter, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours
from test_model import get_continuation_dataset


# for basis, pick one of:
# basis_transform_coords, basis_transform_rouse, basis_transform_neighbours
BASIS = basis_transform_neighbours
SHOW_REALSPACE_SAMPLES = 10


def compare_predictions_x(x_init, x_actual, sim, show_histogram=True):
  x_init = x_init.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  x_actual = x_actual.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  plotter = Plotter(BASIS, samples_subset_size=SHOW_REALSPACE_SAMPLES)
  plotter.plot_samples(x_actual)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  if show_histogram:
    plotter.plot_hist(x_actual)
    plotter.plot_hist_ic(x_init)
    plotter.show()


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
  test_config = Config(sim_nm, "none", x_only=True, t_eql=4)
  init_states, fin_states = get_continuation_dataset(10, 10000, test_config)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  # compare!
  eval_sample_step(init_states, fin_states, test_config.sim)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])







