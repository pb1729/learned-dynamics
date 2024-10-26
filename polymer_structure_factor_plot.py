import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import model_list_to_predictor_list
from config import load, Config
from utils import PrintTiming


def get_x(args, predictor):
  print("GET X")
  state = predictor.sample_q(args.batch)
  print(torch.all(~torch.isnan(state.x)))
  predictor.predict(args.tmax, state)
  print(torch.all(~torch.isnan(state.x)))
  return state.x


def structure_factor(q, x):
  """ structure factor S(q)
      q: (res)
      x: (batch, poly_len, 3)
      S: (res) """
  q = torch.tensor(q, device=x.device)
  delta_x = x[:, :, None] - x[:, None, :] # (batch, poly_len, poly_len, 3)
  S = torch.zeros_like(q)
  for i in range(q.shape[0]): # serial loop to reduce memory consumption
    # S is guaranteed even and real by rotational symmetry, so we just use the cosine here
    S_i = torch.cos(q[i]*delta_x) # (batch, poly_len, poly_len, 3)
    S_i = S_i.mean(-1) # (batch, poly_len, poly_len) to improve stats, we average over q = (1,0,0)q, (0,1,0)q, (0,0,1)q
    S_i = S_i.sum(-1).mean(-1) # (batch) overall factor of (1/N) for both poly_len sums
    S[i] = S_i.mean()
  return S


def main(args):
  predictors = model_list_to_predictor_list(args.models)
  x_list = []
  labels = []
  for predictor in predictors:
    x_list.append(get_x(args, predictor))
    labels.append(predictor.name)
  q = np.exp(np.linspace(np.log(0.15915494309189535/max([x.shape[2] for x in x_list])) - 2.3, np.log(6.283185307179586) + 2.3, 200))
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
  parser.add_argument("tmax", type=int) # positional: we make the user think about how many equililbration steps they want
  parser.add_argument("-M", dest="models", action="extend", nargs="+", type=str)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  main(parser.parse_args())
