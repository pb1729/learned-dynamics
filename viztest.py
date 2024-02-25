import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from sims import sims, get_dataset, subtract_cm_1d, polymer_length
from vampnets import KoopmanModel
from vamp_score import batched_xy_moment


def tica_1(x):
  rowvec = torch.cos(1*np.pi*(torch.arange(polymer_length, device=x.device) + 0.5)/polymer_length)
  return (x*rowvec).sum(-1)
def tica_2(x):
  rowvec = torch.cos(2*np.pi*(torch.arange(polymer_length, device=x.device) + 0.5)/polymer_length)
  return (x*rowvec).sum(-1)


def tica_comparison_plot(ax, eigenfn, idx, tica_1, tica_2, dataset):
    states = dataset[:, -1]
    with torch.no_grad():
      x = tica_1(states)
      y = tica_2(states)
      z = eigenfn(states)[:, idx]
    ax.scatter(x.cpu(), y.cpu(), c=z.cpu())
    #ax.xlabel("tica_1")
    #ax.ylabel("tica_2")


def corr_plot(ax, eigenfn, idx, dataset):
  N, L, dim = dataset.shape
  states = dataset.reshape(N*L, dim)
  states -= states.mean(0, keepdim=True)
  with torch.no_grad():
    corr = batched_xy_moment(states, eigenfn(states))
  ax.set_ylim(-1.3, 1.3)
  ax.scatter(np.arange(dim), corr[:, idx].cpu())


def main(nm, dim):
  dim = int(dim)
  # generate a new dataset for testing
  print("generating polymer dataset...")
  dataset_poly = subtract_cm_1d(get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], 2000, 20))
  print("done.")
  kmod = KoopmanModel.load("models/%s_%d.pt" % (nm, dim))
  sz, = kmod.S.shape
  sidelen = int(np.ceil(np.sqrt(sz)))
  # just plot eigenvalues:
  plt.scatter(np.arange(sz), kmod.S.cpu())
  plt.show()
  # plot the tica1 by tica2 plots
  fig, axs = plt.subplots(sidelen, sidelen)
  for i, sigma in enumerate(kmod.S):
    ax = axs[i // sidelen, i % sidelen]
    tica_comparison_plot(ax, lambda x: kmod.eigenfn_0(x), i, tica_1, tica_2, dataset_poly)
  plt.show()
  # plot strongest linear component plots
  fig, axs = plt.subplots(sidelen, sidelen)
  for i, sigma in enumerate(kmod.S):
    ax = axs[i // sidelen, i % sidelen]
    corr_plot(ax, lambda x: kmod.eigenfn_0(x), i, dataset_poly)
  plt.show()



if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


