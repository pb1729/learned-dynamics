import mdtraj
import numpy as np
import matplotlib.pyplot as plt
from contact_map import ContactFrequency

from managan.backmap_to_pdb import model_state_to_mdtraj
from predictor_argparse_util import args_to_predictor_list, add_model_list_arg


def get_contacts_matrix(md_trajectory):
  frame_contacts = ContactFrequency(md_trajectory, n_neighbors_ignored=1)
  return frame_contacts.residue_contacts.sparse_matrix.todense()

def model_state_contacts(model_state):
  return get_contacts_matrix(model_state_to_mdtraj(model_state))


def main(args):
  predictors = args_to_predictor_list(args)
  pred_name_to_matrix = {}
  for predictor in predictors:
    state = predictor.sample_q(1)
    traj = predictor.predict(args.tmax, state)
    pred_name_to_matrix[predictor.name] = model_state_contacts(traj)
  for name in pred_name_to_matrix:
    print(name)
    plt.imshow(pred_name_to_matrix[name], vmin=0., vmax=1.)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--tmax", type=int, default=100)
  main(parser.parse_args())
