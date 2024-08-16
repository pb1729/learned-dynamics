import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import get_sample_step, get_continuation_dataset, gaussian_kl_div
from config import load


def get_avg_kl(n_iter, model, args):
  # define sampling function
  sample_step = get_sample_step(model)
  def sample_steps(state):
    ans = state.to(torch.float32)
    for i in range(n_iter):
      ans = sample_step(ans)
    return ans.to(torch.float64)
  x0, x1 = get_continuation_dataset(args.samples, args.contins, model.config, iterations=n_iter)
  x1_hat = sample_steps(x0.reshape(-1, model.config.sim.dim)).reshape(-1, args.contins, model.config.sim.dim)
  divs = []
  for i in range(args.samples):
    divs.append(gaussian_kl_div(x1[i], x1_hat[i]))
  divs = np.array(divs)
  div_μ  = divs.mean()
  div_uμ = divs.std()/(args.samples**0.5) # uncertainty in the mean
  return div_μ, div_uμ


def compute_by_iter(args, model):
  ans = []
  for n_iter in args.iter:
    div_μ, div_uμ = get_avg_kl(n_iter, model, args)
    Δt = n_iter*model.config.sim.delta_t
    print("%d, %f, %f, %f" % (n_iter, Δt, div_μ, div_uμ))
    ans.append((Δt, div_μ, div_uμ))
  return ans

def compute_by_trainsteps(args, model):
  assert len(args.iter) == 1, "can't compute by nsteps and iter list simultaneously"
  assert type(model.config.nsteps) == list
  ans = []
  for nsteps in model.config.nsteps[:-1]:
    checkpoint_path = args.fpath.replace(".pt", ".chkp_%d.pt" % nsteps)
    model_chkp = load(checkpoint_path)
    div_μ, div_uμ = get_avg_kl(1, model_chkp, args)
    print("%d, %f, %f" % (nsteps, div_μ, div_uμ))
    ans.append((nsteps, div_μ, div_uμ))
  div_μ, div_uμ = get_avg_kl(1, model, args)
  print("%d, %f, %f" % (model.config.nsteps[-1], div_μ, div_uμ))
  ans.append((model.config.nsteps[-1], div_μ, div_uμ))
  return ans

def main_compute(args):
  # load the model
  model = load(args.fpath)
  model.set_eval(True)
  if args.trainsteps:
    ans = compute_by_trainsteps(args, model)
  else:
    ans = compute_by_iter(args, model)
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
  parser = ArgumentParser(prog="full_model_test_1d")
  parser.add_argument("fpath")
  parser.add_argument("--iter", dest="iter", type=int, action="append", default=[1])
  parser.add_argument("--contins", dest="contins", type=int, default=10000)
  parser.add_argument("--samples", dest="samples", type=int, default=24)
  parser.add_argument("--plot", dest="plot", action="store_true") # plot previously recorded datas
  parser.add_argument("--trainsteps", dest="trainsteps", action="store_true") # get or data for training steps elapsed rather than # iterations
  main(parser.parse_args())


