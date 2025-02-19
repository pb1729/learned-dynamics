import mdtraj
import numpy as np
import matplotlib.pyplot as plt

from managan.backmap_to_pdb import model_state_to_mdtraj
from predictor_argparse_util import args_to_predictor_list, add_model_list_arg


def main(args):
  predictors = args_to_predictor_list(args)
  for predictor in predictors:
    state = predictor.sample_q(1)
    traj = predictor.predict(args.tmax, state)
    mdtrajec = model_state_to_mdtraj(traj)
    _, phis = mdtraj.compute_phi(mdtrajec)
    _, psis = mdtraj.compute_psi(mdtrajec)
    plt.scatter(phis.flatten(), psis.flatten(), label=predictor.name)
  plt.legend()
  plt.xlim(-np.pi, np.pi)
  plt.ylim(-np.pi, np.pi)
  plt.xlabel("φ [rad]")
  plt.ylabel("ψ [rad]")
  plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--tmax", type=int, default=100)
  main(parser.parse_args())
