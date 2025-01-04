import torch
import numpy as np
import matplotlib.pyplot as plt


from test_model import model_list_to_predictor_list
from managan.config import load, Config
from managan.utils import PrintTiming


def get_traj(args, predictor):
  with PrintTiming("sampling initial states with predictor %s" % predictor.name):
    state = predictor.sample_q(args.batch)
  with PrintTiming("evolving forwards with predictor %s" % predictor.name):
    traj = predictor.predict(args.tmax, state)
  return traj.x

def atom_0_msd(x):
  """ mean square displacement of atom 0 in the chain relative to start as a function of time """
  return ((x[:, :, 0] - x[0, None :, 0])**2).sum(-1).mean(1)

def com_msd(x):
  """ mean square displacement of chain center of mass relative to start as a function of time """
  x = x.mean(2)
  return ((x - x[:, 0, None])**2).sum(-1).mean(1)

MSD_FNS = {
  "atom_0": atom_0_msd,
  "com": com_msd,
}

def main(args):
  predictors = model_list_to_predictor_list(args.models)
  x_list = []
  for predictor in predictors:
    x_list.append(get_traj(args, predictor))
  steps = np.arange(args.tmax)
  for x in x_list:
    msd = MSD_FNS[args.axis](x).cpu().numpy()
    plt.plot(steps[1:], msd[1:])
  plt.xscale("log")
  plt.yscale("log")
  if args.slopegrid:
    x = np.exp(np.linspace(0, np.log(args.tmax)))
    y_hlf = x**0.5
    y_min, y_max = plt.ylim()
    for p in range(-3, 4):
      plt.plot(x, y_max*x*(2**p)/x[-1], color="grey", alpha=0.5)
      plt.plot(x, y_min*y_hlf*(2**p), color="grey", alpha=0.5)
  plt.xlabel("t [iterations]")
  plt.ylabel("〈%s^2〉" % args.axis)
  plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="diffusion_plot")
  parser.add_argument("-M", dest="models", action="extend", nargs="+", type=str)
  parser.add_argument("--tmax", dest="tmax", type=int, default=2000)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  parser.add_argument("--slopegrid", dest="slopegrid", action="store_true")
  parser.add_argument("--axis", dest="axis", choices=[key for key in MSD_FNS], default="atom_0")
  main(parser.parse_args())
