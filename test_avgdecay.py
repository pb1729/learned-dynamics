import torch
import numpy as np
import matplotlib.pyplot as plt

from predictor import ModelPredictor
from utils import must_be
from config import get_predictor
from test_model import BASES, model_list_to_predictor_list
from plotting_common import squarish_factorize, basis_transform_coords, basis_transform_rouse, basis_transform_neighbours, basis_transform_neighbours2, basis_transform_neighbours4


def comparison_plot(x_list, shape, args):
  if len(shape) == 2:
    poly_len, space_dim = shape
  else: # just pretend each dim is a 1d atom
    poly_len, = shape
    space_dim = 1
  plt_w, plt_h = squarish_factorize(poly_len)
  fig = plt.figure(figsize=(20, 12))
  axes = [fig.add_subplot(plt_h, plt_w, n + 1) for n in range(poly_len)]
  for i, x in enumerate(x_list):
    batch, tmax, must_be[poly_len], must_be[space_dim] = x.shape
    y = BASES[args.basis](x.cpu().numpy().reshape(batch*tmax, poly_len, space_dim)).reshape(batch, tmax, poly_len, space_dim)
    y_avg = y.mean(0)
    abs_y_avg = np.sqrt((y_avg**2).sum(-1)) # get magnitude of averaged vector
    for j in range(poly_len):
      axes[j].plot(abs_y_avg[:, j])
  for j in range(poly_len):
    _, top = axes[j].get_ylim()
    axes[j].set_ylim(bottom=0., top=1.2*top)
  plt.show()


def get_x_list(state, args, predictors):
  states = []
  for predictor in predictors:
    if isinstance(predictor, ModelPredictor):
      states.append(state.expand(args.contins).to_model_predictor_state())
    else:
      states.append(state.expand(args.contins))
  ans = []
  for state, predictor in zip(states, predictors):
    x = torch.zeros(args.contins, 1 + args.tmax, *predictor.shape)
    x[:, 0] = state.x
    x[:, 1:] = predictor.predict(args.tmax, state)
    ans.append(x)
  return ans


def eval_predictors(args, predictors):
  """ compare a predictor to its base predictor """
  shape = predictors[0].shape
  state = predictors[0].sample_q(1) # get our initial condition from the first predictor in the list
  # evaluate the base predictor and model predictor dynamics
  x_list = get_x_list(state, args, predictors)
  comparison_plot(x_list, shape, args)


def model_list_to_predictor_list(models):
  ans = []
  for model in models:
    predictor = get_predictor(model)
    if isinstance(predictor, ModelPredictor): # add the base predictor for a model first.
      ans.append(predictor.get_base_predictor())
    ans.append(predictor)
  return ans


def main(args):
  print(args)
  predictors = model_list_to_predictor_list(args.models)
  eval_predictors(args, predictors)




if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_avgdecay")
  parser.add_argument("-M", dest="models", action="extend", nargs="+", type=str)
  parser.add_argument("--basis", dest="basis", choices=[key for key in BASES], default="rouse")
  parser.add_argument("--contins", dest="contins", type=int, default=1000)
  parser.add_argument("--tmax", dest="tmax", type=int, default=32)
  main(parser.parse_args())





