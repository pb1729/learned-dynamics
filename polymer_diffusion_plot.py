import torch
import numpy as np
import matplotlib.pyplot as plt


from test_model import get_sample_step
from sims import equilibrium_sample, get_dataset
from config import load, Config
from utils import PrintTiming


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

def atom_0_msd(x):
  """ mean square displacement of atom 0 in the chain relative to start as a function of time """
  return ((x[:, :, 0] - x[:, 0, None, 0])**2).sum(-1).mean(0)

def com_msd(x):
  """ mean square displacement of chain center of mass relative to start as a function of time """
  x = x.mean(2)
  return ((x - x[:, 0, None])**2).sum(-1).mean(0)

MSD_FNS = {
  "atom_0": atom_0_msd,
  "com": com_msd,
}

def main(args):
  x_list = []
  if args.usemodel is not None:
    model = load(args.usemodel)
    model.set_eval(True)
    with PrintTiming("computing model data..."):
      x_g = get_model_x(args, model)
      x_list.append(x_g)
    with PrintTiming("computing simulation data..."):
      x_r = get_sim_x(args, model.config)
      x_list.append(x_r)
  if args.usesim is not None:
    config = Config(args.sim_nm, "none", x_only=True, t_eql=4)
    with PrintTiming("computng simulation data..."):
      x_r = get_sim_x(args, config)
      x_list.append(x_r)
  steps = np.arange(args.tmax)
  for x in x_list:
    msd = MSD_FNS[args.axis](x).cpu().numpy()
    plt.plot(steps[1:], msd[1:])
  plt.xscale("log")
  plt.yscale("log")
  if args.slopegrid:
    x = np.exp(np.linspace(0, np.log(args.tmax)))
    y_hlf = x**0.5
    for p in range(-3, 5):
      plt.plot(x, x*(2**p), color="grey", alpha=0.5)
      plt.plot(x, y_hlf*(2**p), color="grey", alpha=0.5)
  plt.xlabel("t [iterations]")
  plt.ylabel("〈%s^2〉" % args.axis)
  plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="diffusion_plot")
  parser.add_argument("--usesim", dest="usesim", default=None)
  parser.add_argument("--usemodel", dest="usemodel", default=None)
  parser.add_argument("--tmax", dest="tmax", type=int, default=2000)
  parser.add_argument("--batch", dest="batch", type=int, default=256)
  parser.add_argument("--slopegrid", dest="slopegrid", action="store_true")
  parser.add_argument("--axis", dest="axis", choices=[key for key in MSD_FNS], default="atom_0")
  main(parser.parse_args())


