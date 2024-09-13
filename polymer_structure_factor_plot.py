import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import model_list_to_predictor_list
from sims import equilibrium_sample, get_dataset
from config import load, Config
from utils import PrintTiming

# TODO: this file is not up to date with the Predictor/State refactor.
# update it if you want to use it...


def get_sim_x(args, config):
  x_init, v_init = equilibrium_sample(config, args.batch)
  x = get_dataset(config, [x_init, v_init], args.tmax)
  return x.to(torch.float32).reshape(args.batch, args.tmax, config.sim.poly_len, 3)

def get_model_x(args, model):
  sample_step = get_sample_step(model)
  ans = torch.zeros(args.batch, args.tmax, model.config.state_dim, device=model.config.device)
  x, _ = equilibrium_sample(model.config, args.batch)
  x = x.to(torch.float32).to(model.config.device)
  for i in range(args.tmax):
    ans[:, i] = model.predict(x)
    x = ans[:, i]
  return ans.reshape(args.batch, args.tmax, model.config.sim.poly_len, 3)

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
  x_list = []
  labels = []
  for model_path in args.usemodel:
    model_nm = model_path.split("/")[-1]
    model = load(model_path)
    model.set_eval(True)
    with PrintTiming("computing model data for %s" % model_nm):
      x_g = get_model_x(args, model)
      x_list.append(x_g)
      labels.append("model %s" % model_nm)
    with PrintTiming("computing simulation data for %s" % model_nm):
      x_r = get_sim_x(args, model.config)
      x_list.append(x_r)
      labels.append("sim %s" % model.config.sim_name)
  for sim_nm in args.usesim:
    config = Config(sim_nm, "none", x_only=True, t_eql=4)
    with PrintTiming("computing simulation data for %s" % sim_nm):
      x_r = get_sim_x(args, config)
      x_list.append(x_r)
      labels.append("sim %s" % sim_nm)
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
  parser.add_argument("--usesim", dest="usesim", action="append", default=[])
  parser.add_argument("--usemodel", dest="usemodel", action="append", default=[])
  parser.add_argument("--tmax", dest="tmax", type=int, default=8)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  main(parser.parse_args())


