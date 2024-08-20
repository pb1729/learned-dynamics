import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import must_be, batched_model_eval
from config import load
from sims import equilibrium_sample, get_dataset
from plotting_common import Plotter, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours, basis_transform_neighbours2, basis_transform_neighbours4


BASES = {
  "coords": basis_transform_coords,
  "rouse": basis_transform_rouse,
  "neighbours": basis_transform_neighbours,
  "neighbours2": basis_transform_neighbours2,
  "neighbours4": basis_transform_neighbours4,
}
SHOW_REALSPACE_SAMPLES = 10
EVAL_BATCHSZ = 1024


def is_gan(model):
  return hasattr(model, "is_gan") and model.is_gan


def continuation(x_init, v_init, contins, config, iterations=1):
  # expand along another dim to enumerate continuations
  batch,          must_be[config.sim.dim] = x_init.shape
  must_be[batch], must_be[config.sim.dim] = v_init.shape
  x_init = x_init[:, None].expand(batch, contins, config.sim.dim)
  v_init = v_init[:, None].expand(batch, contins, config.sim.dim)
  x = x_init.clone() # will be written to, so clone
  v = v_init.clone() # will be written to, so clone
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


def get_continuation_dataset(batch, contins, config, iterations=1):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...")
  x_init, v_init = equilibrium_sample(config, batch)
  return continuation(x_init, v_init, contins, config, iterations)


def get_sample_step(model):
  """ given a model and current state, predict the next state """
  model.set_eval(True)
  def sample_step(state):
    with torch.no_grad():
      ans = batched_model_eval(
        (lambda x: model.predict(model.config.cond(x))),
        state, model.config.state_dim, batch=EVAL_BATCHSZ)
    return ans
  return sample_step


def compare_predictions_x(x_init, x_predicted, x_actual, sim, basis, show_histogram=True, radial=False):
  if hasattr(sim, "poly_len") and hasattr(sim, "space_dim"):
    poly_len = sim.poly_len
    space_dim = sim.space_dim
  else: # just pretend each dim is a 1d atom
    poly_len = sim.dim
    space_dim = 1
  x_init = x_init.cpu().numpy().reshape(-1, poly_len, space_dim)
  x_predicted = x_predicted.cpu().numpy().reshape(-1, poly_len, space_dim)
  x_actual = x_actual.cpu().numpy().reshape(-1, poly_len, space_dim)
  plotter = Plotter(BASES[basis], samples_subset_size=SHOW_REALSPACE_SAMPLES, title=("basis type = " + basis))
  plotter.plot_samples(x_actual)
  plotter.plot_samples(x_predicted)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  if show_histogram:
    if radial:
      plotter.plot_hist_radial(x_actual)
      plotter.plot_hist_radial(x_predicted)
    else:
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
  cov_actl = torch.cov(x_actual.T).reshape(dim, dim) # we're forced to reshape because the behaviour of cov is inconsistent for vectors of dimension 1
  cov_pred = torch.cov(x_predicted.T).reshape(dim, dim)
  d_mu = mu_pred - mu_actl
  inv_cov_pred = torch.linalg.inv(cov_pred)
  kl_means = 0.5*(d_mu*(inv_cov_pred @ d_mu)).sum()
  kl_covar = 0.5*torch.trace(inv_cov_pred @ cov_actl)
  kl_lgdet = 0.5*(torch.log(torch.det(cov_pred)/torch.det(cov_actl)) - dim)
  return (kl_means + kl_covar + kl_lgdet).item()


def eval_sample_step(sample_step, init_statess, fin_statess, config, basis, radial=False, showkl=False):
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
    if showkl:
      print(gaussian_kl_div(fin_states, pred_fin_states), end="\t")
      print(gaussian_kl_div(fin_states[0::2], fin_states[1::2]))
    compare_predictions_x(init_states[0], pred_fin_states, fin_states, config.sim, basis, radial=radial)


def main(args):
  print(args)
  print("basis = %s    iterations = %d" % (args.basis, args.iter))
  # load the model
  model = load(args.fpath)
  model.set_eval(True)
  # define sampling function
  sample_step = get_sample_step(model)
  def sample_steps(state):
    ans = state
    for i in range(args.iter):
      ans = sample_step(ans)
    return ans
  # get comparison data
  init_states, fin_states = get_continuation_dataset(args.samples, args.contins, model.config, iterations=args.iter)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  # compare!
  if not is_gan(model): # don't do so many samples if we're not distribution matching
    init_states = init_states[:, 0:1]
  eval_sample_step(sample_steps, init_states, fin_states, model.config, args.basis, radial=args.radial, showkl=args.showkl)




if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  parser.add_argument("fpath")
  parser.add_argument("--basis", dest="basis", choices=[key for key in BASES], default="rouse")
  parser.add_argument("--radial", dest="radial", action="store_true")
  parser.add_argument("--iter", dest="iter", type=int, default=1)
  parser.add_argument("--contins", dest="contins", type=int, default=10000)
  parser.add_argument("--samples", dest="samples", type=int, default=4)
  parser.add_argument("--showkl", dest="showkl", action="store_true")
  main(parser.parse_args())






