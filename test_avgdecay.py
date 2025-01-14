import torch
import numpy as np
import matplotlib.pyplot as plt

from managan.predictor import ModelPredictor
from managan.utils import must_be
from managan.config import get_predictor
from test_model import BASES
from predictor_argparse_util import args_to_predictor_list, add_model_list_arg
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
    tmax, batch, must_be[poly_len], must_be[space_dim] = x.shape
    y = BASES[args.basis](x.cpu().numpy().reshape(batch*tmax, poly_len, space_dim)).reshape(tmax, batch, poly_len, space_dim)
    corr = (y*y[0, None]).sum(-1).mean(1)
    for j in range(poly_len):
      axes[j].plot(corr[:, j])
  for j in range(poly_len):
    _, top = axes[j].get_ylim()
    axes[j].set_ylim(bottom=0., top=1.2*top)
  plt.show()


def get_x_list(state, args, predictors):
  states = []
  for predictor in predictors:
    if isinstance(predictor, ModelPredictor):
      states.append(state.expand(1, 0)[0].to_model_predictor_state()) # expand(1, 0)[0] is just to clone the state
    else:
      states.append(state.expand(1, 0)[0]) #  expand(1, 0)[0] is just to clone the state
  ans = []
  for state, predictor in zip(states, predictors):
    x = torch.zeros(1 + args.tmax, args.contins, *state.shape)
    x[0] = state.x
    traj = predictor.predict(args.tmax, state)
    x[1:] = traj.x
    ans.append(x)
  return ans


def eval_predictors(args, predictors):
  """ compare a predictor to its base predictor """
  state = predictors[0].sample_q(args.contins)
  # evaluate the base predictor and model predictor dynamics
  x_list = get_x_list(state, args, predictors)
  comparison_plot(x_list, state.shape, args)


def main(args):
  print(args)
  predictors = args_to_predictor_list(args)
  eval_predictors(args, predictors)




if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_avgdecay")
  add_model_list_arg(parser)
  parser.add_argument("--basis", dest="basis", choices=[key for key in BASES], default="rouse")
  parser.add_argument("--contins", dest="contins", type=int, default=1000)
  parser.add_argument("--tmax", dest="tmax", type=int, default=32)
  main(parser.parse_args())
