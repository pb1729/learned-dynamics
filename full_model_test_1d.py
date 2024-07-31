import torch
import numpy as np
import matplotlib.pyplot as plt

from test_model import get_sample_step, continuation
from config import load


def do_plotting(args, x0, x1, x1_hat):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.hist2d(x0.flatten(), x1.flatten(),     bins=args.res, range=[[-args.range, args.range], [-args.range, args.range]])
  ax2.hist2d(x0.flatten(), x1_hat.flatten(), bins=args.res, range=[[-args.range, args.range], [-args.range, args.range]])
  ax1.set_title("simulation")
  ax2.set_title("model")
  plt.show()

def main(args):
  print(args)
  # load the model
  model = load(args.fpath)
  model.set_eval(True)
  config = model.config
  assert config.sim.dim == 1, "can only make this plot for a 1d system"
  # define sampling function
  sample_step = get_sample_step(model)
  def sample_steps(state):
    ans = state.to(torch.float32)
    for i in range(args.iter):
      ans = sample_step(ans)
    return ans.to(torch.float64)
  x_init = torch.linspace(-args.range, args.range, args.res, device=config.device, dtype=torch.float64).reshape(args.res, 1)
  v_init = torch.zeros_like(x_init)
  x0, x1 = continuation(x_init, v_init, args.contins, config, args.iter)
  x1_hat = sample_steps(x0.reshape(-1, 1)).reshape(-1, args.contins, 1)
  do_plotting(args, x0.cpu().numpy(), x1.cpu().numpy(), x1_hat.cpu().numpy())


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="full_model_test_1d")
  parser.add_argument("fpath")
  parser.add_argument("--iter", dest="iter", type=int, default=1)
  parser.add_argument("--contins", dest="contins", type=int, default=10000)
  parser.add_argument("--range", dest="range", type=float, default=3.)
  parser.add_argument("--res", dest="res", type=int, default=24)
  main(parser.parse_args())


