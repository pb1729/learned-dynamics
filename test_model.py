import torch
import numpy as np
import matplotlib.pyplot as plt

from config import load
from sims import equilibrium_sample, get_dataset
from plotting_common import Plotter, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours


BASES = {
  "coords": basis_transform_coords,
  "rouse": basis_transform_rouse,
  "neighbours": basis_transform_neighbours,
}
SHOW_REALSPACE_SAMPLES = 10
# override number of steps we're looking at
ITERATIONS = 1

def is_gan(model):
  return hasattr(model, "is_gan") and model.is_gan


def get_continuation_dataset(batch, contins, config, iterations=1):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...")
  x_init, v_init = equilibrium_sample(config, batch)
  # expand along another dim to enumerate continuations
  x = x_init[:, None].expand(batch, contins, config.sim.dim).clone() # will be written to, so clone
  v = v_init[:, None].expand(batch, contins, config.sim.dim).clone() # will be written to, so clone
  x = x.reshape(batch*contins, -1)
  v = v.reshape(batch*contins, -1)
  print("created. calculating continuations from initial states...")
  x_fin = get_dataset(config, [x, v], iterations)
  print("done.")
  return x_init, x_fin.reshape(batch, contins, iterations, config.sim.dim)


def get_sample_step(model):
  """ given a model and current state, predict the next state """
  model.set_eval(True)
  def sample_step(state):
    with torch.no_grad():
      state_fin = model.predict(model.config.cond(state))
    return state_fin
  return sample_step


def compare_predictions_x(x_init, x_predicted, x_actual, sim, basis, show_histogram=True):
  x_init = x_init.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  x_predicted = x_predicted.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  x_actual = x_actual.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  plotter = Plotter(BASES[basis], samples_subset_size=SHOW_REALSPACE_SAMPLES, title=("basis type = " + basis))
  plotter.plot_samples(x_actual)
  plotter.plot_samples(x_predicted)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  if show_histogram:
    plotter.plot_hist(x_actual)
    plotter.plot_hist(x_predicted)
    plotter.plot_hist_ic(x_init)
    plotter.show()


def eval_sample_step(sample_step, init_statess, fin_statess, config, basis):
  """ given a method that continues evolution for one more step,
      plot various graphs to evaluate it for accuracy
      sample_step:  (batch, state_dim) -> (batch, state_dim)
      init_statess: (batch, state_dim)
      fin_statess:  (batch, contins, state_dim) """
  batch, *_ = fin_statess.shape
  for i in range(batch):
    init_states = init_statess[i]
    fin_states = fin_statess[i]
    pred_fin_states = sample_step(init_states)
    compare_predictions_x(init_states[0], pred_fin_states, fin_states, config.sim, basis)


def main(fpath, basis="rouse", iterations=1, contins=10000):
  assert fpath is not None
  assert basis in BASES
  # load the model
  model = load(fpath)
  model.set_eval(True)
  # define sampling function
  sample_step = get_sample_step(model)
  def sample_steps(state):
    ans = state
    for i in range(iterations):
      ans = sample_step(ans)
    return ans
  # get comparison data
  init_states, fin_states = get_continuation_dataset(10, contins, model.config, iterations=iterations)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  # compare!
  init_states = init_states[:, None]
  if is_gan(model):
    init_states = init_states.expand(-1, contins, -1)
  eval_sample_step(sample_steps, init_states, fin_states, model.config, basis)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:], iterations=ITERATIONS)






