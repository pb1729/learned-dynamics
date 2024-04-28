import torch
import numpy as np
import matplotlib.pyplot as plt

from config import load
from sims import sims, get_dataset
from polymer_util import rouse, squarish_factorize


ITERATIONS = 1
ROUSE_BASIS = True
SHOW_REALSPACE_SAMPLES = 10



def get_continuation_dataset(N, contins, config, iterations=1):
  """ get a dataset of many possible continued trajectories from each of N initial states """
  print("creating initial states...")
  initial_states = get_dataset(config.sim, N, 1, t_eql=config.t_eql, subtract_cm=config.subtract_mean)[:, 0]
  print("created.")
  # expand along another dim to enumerate continuations
  state_dim = initial_states.shape[-1] # this should include velocity, so we can't use config.state_dim, which may be x only
  xv_init = initial_states[:, None].expand(N, contins, state_dim).clone() # will be written to, so clone
  xv_init = xv_init.reshape(N*contins, state_dim)
  print("calculating continuations from initial states...")
  xv_fin = get_dataset(config.sim, N*contins, iterations, subtract_cm=config.subtract_mean,
    x_init=xv_init[:, :state_dim//2], v_init=xv_init[:, state_dim//2:])[:, -1]
  print("done.")
  xv_fin = xv_fin.reshape(N, contins, state_dim)
  if config.x_only:
    return initial_states[:, :state_dim//2], xv_fin[:, :, :state_dim//2]
  return initial_states, xv_fin


def get_sample_step(model):
  """ given a model and current state, predict the next state """
  model.set_eval(True)
  def sample_step(state):
    batch, _ = state.shape
    with torch.no_grad():
      state_fin = model.predict(model.config.cond(state))
    return state_fin
  return sample_step


def compare_predictions_x(x_init, x_predicted, x_actual, config):
  x_init = x_init.cpu().numpy()
  x_actual = x_actual.cpu().numpy()
  x_predicted = x_predicted.cpu().numpy()
  if SHOW_REALSPACE_SAMPLES > 0:
    plt.plot(x_init[0], marker=".", color="black")
    for i in range(SHOW_REALSPACE_SAMPLES):
      plt.plot(x_actual[i], marker=".", color="blue", alpha=0.7)
    plt.plot(x_predicted[0], marker=".", color="orange", alpha=0.7)
    plt.show()
  fig = plt.figure(figsize=(20, 12))
  axes = []
  plt_w, plt_h = squarish_factorize(config.sim.poly_len)
  for n in range(config.sim.poly_len):
    if ROUSE_BASIS:
      w_x = rouse(n, config.sim.poly_len)[None]
      x_init_n = (w_x*x_init).sum(1)
      x_actual_n = (w_x*x_actual).sum(1)
      x_predicted_n = (w_x*x_predicted).sum(1)
    else:
      x_init_n = x_init[:, n]
      x_actual_n = x_actual[:, n]
      x_predicted_n = x_predicted[:, n]
    axes.append(fig.add_subplot(plt_h, plt_w, n + 1))
    axes[-1].hist(x_actual_n,    range=(-20., 20.), bins=200, alpha=0.7)
    axes[-1].scatter([x_init_n], [0], color="black") # show starting point...
    axes[-1].scatter([x_actual_n.mean()], [0], color="blue")
    axes[-1].scatter([x_predicted_n.mean()], [0], color="orange")
  plt.show()
  #plt.savefig("figures/direct1/%d.png" % np.random.randint(0, 10000))
  #plt.close()


def eval_sample_step(sample_step, init_states, fin_statess, config):
  """ given a method that continues evolution for one more step,
      plot various graphs to evaluate it for accuracy
      sample_step: (batch, state_dim) -> (batch, state_dim)
      init_states: (batch, state_dim)
      fin_statess: (batch, contins, state_dim) """
  batch, contins, _ = fin_statess.shape
  for i in range(batch):
    init_state = init_states[i, None]
    fin_states = fin_statess[i]
    print(init_state.shape)
    pred_fin_states = sample_step(init_state)
    compare_predictions_x(init_state, pred_fin_states, fin_states, config)


def main(fpath, iterations=1):
  assert fpath is not None
  # load the GAN
  model = load(fpath)
  model.set_eval(True)
  # define sampling function
  sample_step = get_sample_step(model)
  def sample_steps(state):
    ans = state
    for i in range(iterations):
      ans = sample_step(ans)
    return ans
  # get comparison data
  init_states, fin_states = get_continuation_dataset(10, 10000, model.config, iterations=iterations)
  init_states, fin_states = init_states.to(torch.float32), fin_states.to(torch.float32)
  print(fin_states.shape, init_states.shape)
  # compare!
  eval_sample_step(sample_steps, init_states, fin_states, model.config)




if __name__ == "__main__":
  from sys import argv
  main(*argv[1:], iterations=ITERATIONS)






