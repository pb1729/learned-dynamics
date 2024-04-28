import torch
import itertools

from sims import dataset_gen
from run_visualization import TensorBoard
from config import Config, Condition, load, save, makenew
from configs import configs


def train(gan, save_path):
  """ train a GAN. inputs:
    gan       - a model to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  print(gan.config)
  board = TensorBoard(run_name)
  config = gan.config # configuration for this run...
  data_generator = dataset_gen(config.sim, config.batch, config.simlen,
    t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only)
  for i in itertools.count():
    trajs = data_generator.send(None if i < 65536 else True)
    if trajs is None: break
    N, L, state_dim = trajs.shape
    encoded_traj = config.cond(trajs.reshape(N*L, state_dim)).reshape(N, L, config.cond_dim)
    cond = encoded_traj[:, :-1].reshape(N*(L - 1), config.cond_dim)
    data = encoded_traj[:,  1:].reshape(N*(L - 1), config.cond_dim)
    loss_d, loss_g = gan.train_step(data, cond)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}\t ℒᴳ = {loss_g:05.6f}")
    board.scalar("loss_d", i, loss_d)
    board.scalar("loss_g", i, loss_g)
    if (i + 1) % 512 == 0:
      print("\nsaving...")
      save(gan, save_path)
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



