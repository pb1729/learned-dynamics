import torch
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import List

from predictor import Predictor, ModelPredictor
from utils import must_be
from config import get_predictor
from plotting_common import Plotter, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours, basis_transform_neighbours2, basis_transform_neighbours4


BASES = {
  "coords": basis_transform_coords,
  "rouse": basis_transform_rouse,
  "neighbours": basis_transform_neighbours,
  "neighbours2": basis_transform_neighbours2,
  "neighbours4": basis_transform_neighbours4,
}
SHOW_REALSPACE_SAMPLES = 10


def comparison_plot(x_init, x_preds, shape, args):
  if len(shape) == 2:
    poly_len, space_dim = shape
  else: # just pretend each dim is a 1d atom
    poly_len, = shape
    space_dim = 1
  x_init = x_init.cpu().numpy().reshape(-1, poly_len, space_dim)
  x_preds = [x_pred.cpu().numpy().reshape(-1, poly_len, space_dim) for x_pred in x_preds]
  plotter = Plotter(BASES[args.basis], samples_subset_size=SHOW_REALSPACE_SAMPLES, title=("basis type = " + args.basis))
  # realspace plot:
  for x_pred in x_preds:
    plotter.plot_samples(x_pred)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  # histogram:
  if args.radial:
    for x_pred in x_preds:
      plotter.plot_hist_radial(x_pred)
  else:
    for x_pred in x_preds:
      plotter.plot_hist(x_pred)
    plotter.plot_hist_ic(x_init)
  plotter.show()


def get_contin_states(state, args, predictors):
  state_contins = []
  for predictor in predictors:
    if isinstance(predictor, ModelPredictor):
      state_contins.append(state.expand(args.contins, 1).to_model_predictor_state())
    else:
      state_contins.append(state.expand(args.contins, 1))
  for state_contin, predictor in zip(state_contins, predictors):
    predictor.predict(args.iter, state_contin, ret=False) # MUTATE state_contins
  return state_contins


def eval_predictors(args, predictors):
  """ compare a predictor to its base predictor """
  state = predictors[0].sample_q(args.samples) # get our set of initial conditions from the first predictor in the list
  # evaluate the base predictor and model predictor dynamics
  state_contins = get_contin_states(state, args, predictors)
  for i in range(args.samples):
    x_init = state[i].x
    x_preds = [state_contin.x[i::args.samples] for state_contin in state_contins]
    comparison_plot(x_init, x_preds, state[i].shape, args)


def model_list_to_predictor_list(models) -> List[Predictor]:
  ans = []
  for model in models:
    predictor:Predictor = get_predictor(model)
    if isinstance(predictor, ModelPredictor): # add the base predictor for a model first.
      ans.append(predictor.get_base_predictor())
    ans.append(predictor)
  return ans


def main(args):
  print(args)
  print("basis = %s    iterations = %d" % (args.basis, args.iter))
  predictors = model_list_to_predictor_list(args.models)
  eval_predictors(args, predictors)




if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  parser.add_argument("-M", dest="models", action="extend", nargs="+", type=str)
  parser.add_argument("--basis", dest="basis", choices=[key for key in BASES], default="rouse")
  parser.add_argument("--radial", dest="radial", action="store_true")
  parser.add_argument("--iter", dest="iter", type=int, default=1)
  parser.add_argument("--contins", dest="contins", type=int, default=10000)
  parser.add_argument("--samples", dest="samples", type=int, default=4)
  main(parser.parse_args())
