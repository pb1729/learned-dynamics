import torch
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from plotting_common import Plotter, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours
from test_model import get_continuation_dataset


BASES = {
  "coords": basis_transform_coords,
  "rouse": basis_transform_rouse,
  "neighbours": basis_transform_neighbours,
}
SHOW_REALSPACE_SAMPLES = 10


def compare_predictions_x(x_init, x_actual, sim, basis, show_histogram=True):
  x_init = x_init.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  x_actual = x_actual.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  plotter = Plotter(BASES[basis], samples_subset_size=SHOW_REALSPACE_SAMPLES, title=("basis type = " + basis))
  plotter.plot_samples(x_actual)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  if show_histogram:
    plotter.plot_hist(x_actual)
    plotter.plot_hist_ic(x_init)
    plotter.show()


def eval_sample_step(init_states, fin_statess, sim, basis):
  batch, contins, _ = fin_statess.shape
  for i in range(batch):
    print("\nnext i.c: i=%d\n" % i)
    init_state = init_states[i, None]
    fin_states = fin_statess[i, :]
    compare_predictions_x(init_state, fin_states, sim, basis)


def atoms_display(config):
  from atoms_display import launch_atom_display
  import time
  from sims import equilibrium_sample
  assert config.sim.space_dim == 3
  def clean_for_display(x):
    return x.cpu().numpy()
  x, v = equilibrium_sample(config, 1)
  display = launch_atom_display(5*np.ones(config.sim.poly_len, dtype=int),
    clean_for_display(x.reshape(config.sim.poly_len, config.sim.space_dim)))
  while True:
    x_traj, v_traj = config.sim.generate_trajectory(x, v, 256)
    x, v = x_traj[:, -1], v_traj[:, -1]
    x_snapshots = x_traj.reshape(1, 256, config.sim.poly_len, config.sim.space_dim)
    for i in range(256):
      time.sleep(0.1)
      display.update_pos(clean_for_display(x_snapshots[0, i]))
    
      
def main(sim_nm, basis="rouse", do_atoms_display=None):
  assert basis in BASES
  # get comparison data
  test_config = Config(sim_nm, "none", x_only=True, t_eql=4)
  if do_atoms_display is not None:
    atoms_display(test_config)
  init_states, fin_states = get_continuation_dataset(10, 1000, test_config)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  # compare!
  eval_sample_step(init_states, fin_states, test_config.sim, basis)

if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])







