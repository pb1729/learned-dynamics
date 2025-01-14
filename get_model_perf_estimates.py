import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import get_contin_states
from predictor_argparse_util import model_list_to_predictor_list
from managan.config import load
from managan.utils import must_be, prod


def gaussian_kl_div(x_actual, x_predicted):
  """ approximate given distributions as Gaussians with the same mean and standard deviation.
      returns the KL divergence between the distributions, the expectation being taken over x_actual.
      https://ra1ndeer.github.io/posts/kl_divergence_gaussians.html
      x_actual:    (samples, ...)
      x_predicted: (samples, ...) """
  samples,          *rest           = x_actual.shape
  must_be[samples], *must_be[rest]  = x_predicted.shape
  dim = prod(rest)
  x_actual = x_actual.reshape(samples, dim)
  x_predicted = x_predicted.reshape(samples, dim)
  with torch.no_grad():
    mu_actl = x_actual.mean(0)
    mu_pred = x_predicted.mean(0)
    d_mu = mu_pred - mu_actl
    # do SVD's because it is more numerically stable than covariances
    _, S_actl, Vh_actl = torch.linalg.svd(x_actual, full_matrices=False)
    _, S_pred, Vh_pred = torch.linalg.svd(x_predicted, full_matrices=False)
    S_pred_inv = torch.diag(samples/(S_pred**2))
    S_actl = torch.diag((S_actl**2)/samples) # square and normalize to get covariance
    S_pred = torch.diag((S_pred**2)/samples) # square and normalize to get covariance
    # compute covariance terms:
    kl_means = 0.5*(d_mu*(Vh_pred.T @ S_pred_inv @ Vh_pred @ d_mu)).sum()
    kl_covar = 0.5*torch.trace(Vh_pred.T @ S_pred_inv @ Vh_pred @ Vh_actl.T @ S_actl @ Vh_actl)
    kl_lgdet = 0.5*(torch.log(torch.diag(S_pred)).sum() - torch.log(torch.diag(S_actl)).sum())
    return (kl_means + kl_covar + kl_lgdet - 0.5*dim).item()


def get_avg_kl(args, predictors):
  # we expect exactly 2 predictors in the list
  state = predictors[0].sample_q(args.samples)
  state_contins = get_contin_states(state, args, predictors)
  divs = []
  for i in range(args.samples):
    divs.append(gaussian_kl_div(state_contins[0].x[i::args.samples], state_contins[1].x[i::args.samples]))
  print(divs)
  divs = np.array(divs)
  div_μ  = divs.mean()
  div_uμ = divs.std()/(args.samples**0.5) # uncertainty in the mean
  return div_μ, div_uμ


def compute_by_iter(args, predictors):
  ans = []
  for n_iter in args.iter:
    args.iter = n_iter
    div_μ, div_uμ = get_avg_kl(args, predictors)
    Δt = n_iter*predictors[0].sim.delta_t # predictors[0] should be a SimPredictor
    print("%d, %f, %f, %f" % (n_iter, Δt, div_μ, div_uμ))
    ans.append((Δt, div_μ, div_uμ))
  return ans

def compute_by_trainsteps(args, predictors):
  assert len(args.iter) == 1, "can't compute by nsteps and iter list simultaneously"
  args.iter = args.iter[0] # single value
  config = predictors[1].model.config # predictors[1] should be a ModelPredictor
  assert type(config.nsteps) == list
  ans = []
  for nsteps in config.nsteps[:-1]:
    if nsteps == 0: continue # some models have 0 as first checkpoint, but this does not get saved
    checkpoint_path = args.fpath.replace(".pt", ".chkp_%d.pt" % nsteps)
    chkp_predictors = model_list_to_predictor_list(["model:"+checkpoint_path])
    div_μ, div_uμ = get_avg_kl(args, chkp_predictors)
    print("%d, %f, %f" % (nsteps, div_μ, div_uμ))
    ans.append((nsteps, div_μ, div_uμ))
  div_μ, div_uμ = get_avg_kl(args, predictors)
  print("%d, %f, %f" % (config.nsteps[-1], div_μ, div_uμ))
  print()
  ans.append((config.nsteps[-1], div_μ, div_uμ))
  return ans

def main_compute(args):
  # load the model
  predictors = model_list_to_predictor_list(["model:"+args.fpath])
  if args.trainsteps:
    ans = compute_by_trainsteps(args, predictors)
  else:
    ans = compute_by_iter(args, predictors)
  print("--- Results: ---")
  for tup in ans:
    print("%f, %f, %f" % tup)
  print()


def get_lines():
  """ get multiline input, stopping when an empty line is submitted or CTL-D is typed """
  ans = []
  while True:
    try:
      line = input()
    except EOFError:
      break
    if line == "": break
    ans.append(line)
  return ans

def main_plot(args):
  ntraces = int(args.fpath) # put a number for fpath in this usage
  traces_x = []
  traces_y = []
  traces_u = []
  for _ in range(ntraces):
    trace_x, trace_y, trace_u = [], [], []
    print("\npaste model perf estimates:\n")
    lines = get_lines()
    for line in lines:
      x, y, u = line.split(", ")
      x, y, u = float(x), float(y), float(u)
      trace_x.append(x)
      trace_y.append(y)
      trace_u.append(u)
    traces_x.append(np.array(trace_x))
    traces_y.append(np.array(trace_y))
    traces_u.append(np.array(trace_u))
  for x, y, u in zip(traces_x, traces_y, traces_u):
    if args.trainsteps:
      label = None
    else:
      label = "Δt = %f" % x[0] # TODO: assumption! is x[0] really for 1 iteration?
    plt.errorbar(x, y, yerr=u, marker="o", label=label)
  plt.xscale("log")
  plt.xlabel("# training steps" if args.trainsteps else "Δt * #iterations")
  plt.ylabel("fake KL div.")
  plt.legend()
  plt.show()

def main(args):
  print(args)
  if args.plot:
    main_plot(args)
  else:
    main_compute(args)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="model_perf_estimates")
  parser.add_argument("fpath")
  parser.add_argument("--iter", dest="iter", type=int, nargs="+", default=[1])
  parser.add_argument("--contins", dest="contins", type=int, default=10000)
  parser.add_argument("--samples", dest="samples", type=int, default=24)
  parser.add_argument("--plot", dest="plot", action="store_true") # plot previously recorded datas
  parser.add_argument("--trainsteps", dest="trainsteps", action="store_true") # get or data for training steps elapsed rather than # iterations
  main(parser.parse_args())
