import torch
import itertools

from sims import dataset_gen
from run_visualization import TensorBoard
from config import Config, Condition, load, save, makenew
from configs import configs


N_TRSTEPS = 2048 # 65536
SAVE_EVERY = 512
assert N_TRSTEPS % SAVE_EVERY == 0

def train(model, save_path):
  """ train a GAN. inputs:
    gan       - a model to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  print(model.config)
  board = TensorBoard(run_name)
  config = model.config # configuration for this run...
  data_generator = dataset_gen(config.sim, config.batch, config.simlen,
    t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only)
  for i in itertools.count():
    trajs = data_generator.send(None if i < N_TRSTEPS else True)
    if trajs is None: break
    N, L, state_dim = trajs.shape
    cond = config.cond(trajs[:, :-1].reshape(N*(L - 1), state_dim))
    data = trajs[:, 1:].reshape(N*(L - 1), state_dim)
    loss = model.train_step(data, cond)
    print(f"{i}\t ℒ = {loss:05.6f}")
    board.scalar("loss", i, loss)
    if (i + 1) % SAVE_EVERY == 0:
      print("\nsaving...")
      save(model, save_path)
      print("saved.\n")


def training_run(save_path, src):
  if isinstance(src, Config): # create new from config
    model = makenew(src)
  elif isinstance(src, str): # load from path
    model = load(src)
  train(model, save_path)


def main(save_path, src):
  if src in configs:
    src = configs[src]
  training_run(save_path, src)

if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])



