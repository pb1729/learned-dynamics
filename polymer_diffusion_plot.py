import torch
import numpy as np
import matplotlib.pyplot as plt


from test_model import get_sample_step
from sims import equilibrium_sample, get_dataset
from config import load


def get_sim_x(args, model):
  x_init, v_init = equilibrium_sample(model.config, args.batch)
  x = get_dataset(model.config, [x_init, v_init], args.tmax)
  return x.to(torch.float32).reshape(args.batch, args.tmax, model.config.sim.poly_len, 3)

def get_model_x(args, model):
  sample_step = get_sample_step(model)
  ans = torch.zeros(args.batch, args.tmax, model.config.state_dim, device=model.config.device)
  x, _ = equilibrium_sample(model.config, args.batch)
  x = x.to(torch.float32).to(model.config.device)
  for i in range(args.tmax):
    ans[:, i] = model.predict(x)
    x = ans[:, i]
  return ans.reshape(args.batch, args.tmax, model.config.sim.poly_len, 3)

def atom_0_msd(x):
  """ mean square displacement of atom 0 in the chain relative to start as a function of time """
  return ((x[:, :, 0] - x[:, 0, None, 0])**2).sum(-1).mean(0)

def main(args):
  model = load(args.fpath)
  model.set_eval(True)
  print("computing simulation data")
  x_r = get_sim_x(args, model)
  print("done. computing model data")
  x_g = get_model_x(args, model)
  print("done.")
  steps = np.arange(args.tmax)
  msd_r = atom_0_msd(x_r).cpu().numpy()
  msd_g = atom_0_msd(x_g).cpu().numpy()
  # plotting
  plt.plot(steps[1:], msd_r[1:])
  plt.plot(steps[1:], msd_g[1:])
  plt.xscale("log")
  plt.yscale("log")
  if args.slopegrid:
    x = np.exp(np.linspace(0, np.log(args.tmax)))
    y_hlf = x**0.5
    for p in range(-3, 5):
      plt.plot(x, x*(2**p), color="grey", alpha=0.5)
      plt.plot(x, y_hlf*(2**p), color="grey", alpha=0.5)
  plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="diffusion_plot")
  parser.add_argument("fpath")
  parser.add_argument("--tmax", dest="tmax", type=int, default=2000)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  parser.add_argument("--slopegrid", dest="slopegrid", action="store_true")
  main(parser.parse_args())


