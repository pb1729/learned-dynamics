import torch
import numpy as np
import matplotlib.pyplot as plt

from predictor_argparse_util import args_to_predictor_list, add_model_list_arg


def get_rms_diffusion(traj):
  """ traj: (L, batch, atoms, 3) """
  return np.sqrt(((traj - traj[0, None])**2).sum(-1).mean(-1).mean(-1))


def comparison_plot(args, trajs, names):
  for traj, name in zip(trajs, names):
    plt.plot(get_rms_diffusion(traj), label=name)
  plt.legend()
  plt.xlabel("t [steps]")
  plt.ylabel("sqrt(<(x_0 - x_t)^2>)")
  if args.logscale:
    plt.xscale("log")
    plt.yscale("log")
  plt.show()


def eval_predictors(args, predictors):
  """ compare a predictor to its base predictor """
  trajs = []
  names = []
  for predictor in predictors:
    traj = []
    for i in range(args.batch):
      print(i)
      state = predictor.sample_q(1)
      traj_i = predictor.predict(args.tmax, state)
      traj.append(traj_i.x_npy)
    trajs.append(np.concatenate(traj, axis=1))
    names.append(predictor.name)
  comparison_plot(args, trajs, names)


def main(args):
  predictors = args_to_predictor_list(args)
  eval_predictors(args, predictors)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_diffuse")
  add_model_list_arg(parser)
  parser.add_argument("--batch", dest="batch", type=int, default=1)
  parser.add_argument("--tmax", dest="tmax", type=int, default=10)
  parser.add_argument("--logscale", action="store_true")
  main(parser.parse_args())
