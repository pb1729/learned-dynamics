import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import must_be
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
  xv_fin = get_dataset(config, [x, v], iterations)[:, -1]
  if config.x_only:
    xv_init = x_init
  else:
    xv_init = torch.cat([x_init, v_init], dim=1)
  print("done.")
  return xv_init, xv_fin.reshape(batch, contins, config.state_dim)


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


def gaussian_kl_div(x_actual, x_predicted):
  """ approximate given distributions as Gaussians with the same mean and standard deviation.
      returns the KL divergence between the distributions, the expectation being taken over x_actual.
      https://ra1ndeer.github.io/posts/kl_divergence_gaussians.html
      x_actual:    (samples, dim)
      x_predicted: (samples, dim) """
  samples,          dim          = x_actual.shape
  must_be[samples], must_be[dim] = x_predicted.shape
  mu_actl = x_actual.mean(0)
  mu_pred = x_predicted.mean(0)
  cov_actl = torch.cov(x_actual.T)
  cov_pred = torch.cov(x_predicted.T)
  d_mu = mu_pred - mu_actl
  inv_cov_pred = torch.linalg.inv(cov_pred)
  kl_means = 0.5*(d_mu*(inv_cov_pred @ d_mu)).sum()
  kl_covar = 0.5*torch.trace(inv_cov_pred @ cov_actl)
  kl_lgdet = 0.5*(torch.log(torch.det(cov_pred)/torch.det(cov_actl)) - dim)
  return (kl_means + kl_covar + kl_lgdet).item()


def eval_sample_step(sample_step, init_statess, fin_statess, config, basis):
  """ given a method that continues evolution for one more step,
      plot various graphs to evaluate it for accuracy
      sample_step:  (contins, state_dim) -> (contins, state_dim)
      init_statess: (batch, state_dim)
      fin_statess:  (batch, contins, state_dim) """
  batch, *_ = fin_statess.shape
  for i in range(batch):
    init_states = init_statess[i]
    fin_states = fin_statess[i]
    pred_fin_states = sample_step(init_states)
    # just plot the x part:
    if not config.x_only:
      init_states = init_states[:, :config.sim.dim]
      fin_states = fin_states[:, :config.sim.dim]
      pred_fin_states = pred_fin_states[:, :config.sim.dim]
    print("Gaussian approximation KL divergence for this instance:", gaussian_kl_div(fin_states, pred_fin_states))
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
  # compare!
  init_states = init_states[:, None]
  if is_gan(model):
    init_states = init_states.expand(-1, contins, -1)
  eval_sample_step(sample_steps, init_states, fin_states, model.config, basis)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:], iterations=ITERATIONS)






