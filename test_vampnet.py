import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from config import load
from sims import sims, get_dataset
from polymer_util import rouse, tica_theory, get_n_quanta_theory


def tica_comparison_plot(eigenfn, idx, tica_1, tica_2, dataset):
  states = dataset[:, -1]
  x = tica_1(states)
  y = tica_2(states)
  z = eigenfn(states)[:, idx]
  plt.scatter(x, y, c=z)
  plt.xlabel("tica_1")
  plt.ylabel("tica_2")
  plt.show()


def main(path, abs_scale=False):
  # load the model
  model = load(path)
  config = model.config
  polymer_length = config.sim.poly_len
  n_theory = config["outdim"] + 1
  # plotting setup
  x = np.arange(1, polymer_length)
  plt.scatter(x, tica_theory(config.sim), color="black", marker="o", facecolors="none", label="theory (up to 1 quanta)")
  plt.scatter(np.arange(n_theory), get_n_quanta_theory(n_theory, config.sim), color="black", marker="o", label="theory (any # of quanta)")
  # generate a new dataset for testing
  print("generating polymer dataset...")
  dataset = get_dataset(config.sim, 6000, config.simlen, t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only).to(torch.float32)
  print("done.")
  # evaluate model
  scores = model.eval_score(dataset)
  sdim, = model.S.shape
  plt.scatter(np.arange(1, 1 + sdim),  scores.cpu().numpy(), marker="o", label="model evaluation score")
  plt.scatter(np.arange(1, 1 + sdim), model.S.cpu().numpy(), marker="o", label="learned S")
  if abs_scale:
    plt.ylim(0., 1.)
  plt.xlabel("eigenvalue index")
  plt.ylabel("eigenvalue")
  plt.legend()
  plt.show()
  n_tica12_plots = int(input("how many tica1 x tica2 plots should we show? "))
  # do comparison with tica1 and tica2 plots
  rouse1, rouse2 = torch.tensor(rouse(1, polymer_length)).to("cuda", torch.float32), torch.tensor(rouse(2, polymer_length)).to("cuda", torch.float32)
  def tica1(x):
    with torch.no_grad():
      return (x @ rouse1).cpu().numpy()
  def tica2(x):
    with torch.no_grad():
      return (x @ rouse2).cpu().numpy()
  for i in range(n_tica12_plots):
    def eigenfn(x):
      with torch.no_grad():
        return model.eigenfn_0(x).cpu().numpy()
    tica_comparison_plot(eigenfn, i, tica1, tica2, dataset)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])




