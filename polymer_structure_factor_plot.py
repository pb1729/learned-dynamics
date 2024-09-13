import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import model_list_to_predictor_list
from config import load, Config
from utils import PrintTiming


def get_x(args, predictor):
  state = predictor.sample_q(args.batch)
  return predictor.predict(args.tmax, state)

def structure_factor(q, x):
  """ structure factor S(q)
      q: (res)
      x: (batch, tmax, poly_len, 3) """
  q = torch.tensor(q, device=x.device)[:, None, None, None, None, None]
  delta_x = x[:, :, :, None] - x[:, :, None, :]
  delta_x = delta_x[None] # (1, batch, tmax, poly_len, poly_len, 3)
  # S is guaranteed even and real by rotational symmetry, so we just use the cosine here
  # to improve stats, we average over q = (1,0,0)q, (0,1,0)q, (0,0,1)q
  S = torch.cos(q*delta_x) # (res, batch, tmax, poly_len, poly_len, 3)
  S = S.mean(-1) # (res, batch, tmax, poly_len, poly_len)
  S = S.sum(-1).mean(-1) # (res, batch, tmax) overall factor of (1/N) for both poly_len sums
  S = S.mean((1, 2)) # (res)
  return S


def main(args):
  predictors = model_list_to_predictor_list(args.models)
  x_list = []
  labels = []
  for predictor in predictors:
    x_list.append(get_x(args, predictor))
    labels.append(predictor.name)
  q = np.exp(np.linspace(np.log(0.15915494309189535/max([x.shape[2] for x in x_list])) - 2.3, np.log(6.283185307179586) + 2.3))
  for x, label in zip(x_list, labels):
    S = structure_factor(q, x).cpu().numpy()
    plt.plot(q, S, label=label)
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("q")
  plt.ylabel("S(q)")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="structure_factor_plot")
  parser.add_argument("-M", dest="models", action="extend", nargs="+", type=str)
  parser.add_argument("--tmax", dest="tmax", type=int, default=8)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  main(parser.parse_args())


