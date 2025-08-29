import mdtraj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from managan.backmap_to_pdb import model_states_to_mdtrajs
from managan.utils import must_be
from plotting.predictor_argparse_util import args_to_predictor_list, add_model_list_arg
from plotting.plotting_common import approx_squarish_factorize


# set big font size
matplotlib.rc("font", size=16)


def lagdiff(x, offset_max=None):
  """ x: (L, batch, atoms, 3)
      ans: (offset_max) """
  if offset_max is None: offset_max = x.shape[0] - 1
  return np.stack([
    0.
  ] + [
    ((x[offset:] - x[:-offset])**2).sum(-1).mean()
    for offset in range(1, offset_max)
  ])


def diffusion_plot(args, trajs):
  # phi hist
  for i, prednm in enumerate(trajs):
    print(f"predictor {i} == {prednm}")
    msd = lagdiff(trajs[prednm], offset_max=args.offset_max)
    plt.plot(msd, label=f"predictor {i}")
  plt.legend()
  plt.xlabel("offset time [τ]")
  plt.ylabel("mean square displacement [Å²]")
  plt.show()


def get_data_for_predictor(args, predictor):
  trajs = []
  for i in range(args.resample):
    state = predictor.sample_q(args.batch)
    traj = predictor.predict(args.tmax, state)
    trajs.append(traj.x_npy)
  return np.concatenate(trajs, axis=1) # concat along batch dim, creating a larger effective batch



def main(args):
  predictors = args_to_predictor_list(args)
  angles = {}
  for predictor in predictors:
    print(predictor.name)
    angles[predictor.name] = get_data_for_predictor(args, predictor)
    print("done")
  diffusion_plot(args, angles)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--resample", type=int, default=1, help="number of times to restart from a new state")
  parser.add_argument("--batch", type=int, default=1, help="batch size")
  parser.add_argument("--tmax", type=int, default=100, help="trajectory length to request")
  parser.add_argument("--offset_max", type=int, default=None, help="maximum offset length to plot")
  main(parser.parse_args())
