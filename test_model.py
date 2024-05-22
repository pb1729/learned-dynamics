import torch
import numpy as np
import matplotlib.pyplot as plt

from config import load
from sims import sims, get_dataset
from plotting_common import Plotter, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours


# for basis, pick one of:
# basis_transform_coords, basis_transform_rouse, basis_transform_neighbours
BASIS = basis_transform_rouse
SHOW_REALSPACE_SAMPLES = 10
# override number of steps we're looking at
ITERATIONS = 1


def is_gan(model):
  return hasattr(model, "get_latents")


def get_continuation_dataset(N, contins, config, iterations=1):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...")
  initial_states = get_dataset(config.sim, N, 1, t_eql=config.t_eql, subtract_cm=config.subtract_mean)[:, 0]
  print("created.")
  # expand along another dim to enumerate continuations
  state_dim = initial_states.shape[-1] # this should include velocity, so we can't use config.state_dim, which may be x only
  xv_init = initial_states[:, None].expand(N, contins, state_dim).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = get_dataset(config.sim, N*contins, iterations, subtract_cm=config.subtract_mean,
    x_init=xv_init[:, :state_dim//2], v_init=xv_init[:, state_dim//2:])[:, -1]
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, state_dim)
  if config.x_only:
    return initial_states[:, :state_dim//2], xv_fin[:, :, :state_dim//2]
  return initial_states, xv_fin


def get_sample_step(model):
  """ given a model and current state, predict the next state """
  model.set_eval(True)
  if is_gan(model):
    def sample_step(state):
      batch, _ = state.shape
      latents = model.get_latents(batch)
      with torch.no_grad():
        state_fin = model.gen(latents, model.config.cond(state))
      return state_fin
    return sample_step
  else:
    def sample_step(state):
      with torch.no_grad():
        state_fin = model.predict(model.config.cond(state))
      return state_fin
    return sample_step


def compare_predictions_x(x_init, x_predicted, x_actual, sim, show_histogram=True):
  x_init = x_init.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  x_predicted = x_predicted.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  x_actual = x_actual.cpu().numpy().reshape(-1, sim.poly_len, sim.space_dim)
  plotter = Plotter(BASIS, samples_subset_size=SHOW_REALSPACE_SAMPLES)
  plotter.plot_samples(x_actual)
  plotter.plot_samples(x_predicted)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  if show_histogram:
    plotter.plot_hist(x_actual)
    plotter.plot_hist(x_predicted)
    plotter.plot_hist_ic(x_init)
    plotter.show()


def eval_sample_step(sample_step, init_statess, fin_statess, config):
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
    compare_predictions_x(init_states[0], pred_fin_states, fin_states, config.sim)


def main(fpath, iterations=1, contins=10000):
  assert fpath is not None
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
  eval_sample_step(sample_steps, init_states, fin_states, model.config)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:], iterations=ITERATIONS)






