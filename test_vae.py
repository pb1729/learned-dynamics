import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import load
from sims import sims, get_dataset
from polymer_util import rouse


SHOW_REALSPACE_SAMPLES = 1


def make_dataset(N, config):
  """ get N trajectories """
  return get_dataset(config.sim, N, config.simlen,
    t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only)


def get_reencode(model):
  """ given a model and current state, encode and decode that state """
  # TODO model.set_eval(True) # TODO: implement in VAE
  def reencode(state):
    return model.decode(model.encode(state))
  return reencode


def compare_encoded_x(x_actual, x_encoded):
  x_actual = x_actual.cpu().numpy()
  x_encoded = x_encoded.cpu().numpy()
  if SHOW_REALSPACE_SAMPLES > 0:
    for i in range(SHOW_REALSPACE_SAMPLES):
      plt.plot(x_actual[i], marker=".", color="blue", alpha=0.7)
      plt.plot(x_encoded[i], marker=".", color="orange", alpha=0.7)
    plt.show()
  for rouse_basis in [False, True]:
    fig = plt.figure(figsize=(20, 12))
    axes = []
    for n in range(12): # TODO: leave polymer length as a variable???
      if rouse_basis:
        w_x = rouse(n, 12)[None]
        x_actual_n = (w_x*x_actual).sum(1)
        x_encoded_n = (w_x*x_encoded).sum(1)
      else:
        x_actual_n = x_actual[:, n]
        x_encoded_n = x_encoded[:, n]
      axes.append(fig.add_subplot(3, 4, n + 1))
      axes[-1].hist(x_actual_n,  range=(-20., 20.), bins=200, alpha=0.7)
      axes[-1].hist(x_encoded_n, range=(-20., 20.), bins=200, alpha=0.7)
      axes[-1].scatter([x_actual_n.mean()], [0], color="blue")
      axes[-1].scatter([x_encoded_n.mean()], [0], color="orange")
    plt.show()
    #plt.savefig("figures/direct1/%d.png" % np.random.randint(0, 10000))
    #plt.close()


def eval_reencode(reencode, x_actual):
  x_encoded = reencode(x_actual)
  compare_encoded_x(x_actual, x_encoded)


def main(fpath):
  assert fpath is not None
  N = 10000
  # load the VAE
  vae = load(fpath)
  # define sampling function
  reencode = get_reencode(vae)
  # get comparison data
  dataset = make_dataset(N, vae.config).reshape(N*vae.config.simlen, -1).to(torch.float32)
  # compare!
  eval_reencode(reencode, dataset)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])






