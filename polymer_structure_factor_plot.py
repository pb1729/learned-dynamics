import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import model_list_to_predictor_list
from managan.config import load, Config
from managan.utils import PrintTiming, avg_relative_diff


def get_x(args, predictor):
  ans = []
  for j in range(0, args.batch, 128):
    state = predictor.sample_q(min(128, args.batch - j))
    predictor.predict(args.tmax, state)
    ans.append(state.x)
  return torch.cat(ans, dim=0)


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


def structure_factor_plot(args, predictors, x_list, labels):
  for predictor, x, label in zip(predictors, x_list, labels):
    box = predictor.get_box()
    if box is None:
      q = np.exp(np.linspace(np.log(0.15915494309189535/max([x.shape[2] for x in x_list])) - 2.3, np.log(6.283185307179586) + 2.3, args.res))
    else:
      assert box[0] == box[1] == box[2], "structure factor plot for uneven box dimensions not supported"
      # pick only q values consistent with periodicity
      q0 = (2*np.pi/box[0]).item()
      qmax_q0 = int(63/q0)
      q = q0*np.arange(1, qmax_q0)
    S = structure_factor(q, x).cpu().numpy()
    if box is None:
      plt.plot(q, S, label=label)
    else:
      plt.scatter(q, S, label=label)
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("q")
  plt.ylabel("S(q)")
  plt.legend()
  plt.show()


def rdf_plot(args, predictors, x_list, labels):
  for predictor, x, label in zip(predictors, x_list, labels):
    delta_x = x[:, :, None] - x[:, None, :] # (batch, nodes, nodes, 3)
    box = predictor.get_box()
    if box is not None:
      rmax = (box.min()/2).item()
      delta_x =(delta_x + box/2)%box - box/2
    else:
      rmax = torch.sqrt((delta_x**2).sum(-1).max()).item()
    r = torch.sqrt((delta_x**2).sum(-1).flatten()).cpu().numpy()
    plt.hist(r, bins=args.res, range=(0., rmax), label=label, weights=1/r**2, alpha=0.5) # weight by 1/r^2 because shell area
  plt.legend()
  plt.show()


def main(args):
  predictors = model_list_to_predictor_list(args.models)
  x_list = []
  labels = []
  for predictor in predictors:
    x_list.append(get_x(args, predictor))
    labels.append(predictor.name)
  if args.rdf:
    rdf_plot(args, predictors, x_list, labels)
  else:
    structure_factor_plot(args, predictors, x_list, labels)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="structure_factor_plot")
  parser.add_argument("tmax", type=int) # positional: we make the user think about how many equililbration steps they want
  parser.add_argument("-M", dest="models", action="extend", nargs="+", type=str)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  parser.add_argument("--rdf", dest="rdf", action="store_true")
  parser.add_argument("--res", dest="res", type=int, default=100)
  main(parser.parse_args())
