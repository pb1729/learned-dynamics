import torch

from sims import dataset_gen
from run_visualization import TensorBoard
from config import Config, Condition, load, save, makenew
from configs import configs


def batchify(dataset_gen, batchsz):
  for dataset in dataset_gen:
    N, L, state_dim = dataset.shape
    assert N % batchsz == 0
    for i in range(0, N, batchsz):
      yield dataset[i:i+batchsz]


def train(vae, save_path):
  """ train a GAN. inputs:
    vae       - a model to be fed training data
    save_path - string, location where the model should be saved to
    board     - None, or a TensorBoard to record training progress """
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  print(vae.config)
  board = TensorBoard(run_name)
  config = vae.config # configuration for this run...
  data_generator = dataset_gen(config.sim, 128*config.batch, config.simlen,
    t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only)
  for i, trajs in enumerate(batchify(data_generator, config.batch)):
    N, L, state_dim = trajs.shape
    data = trajs.reshape(N*L, state_dim)
    loss = vae.train_step(data)
    print(f"{i}\t ℒ = {loss:05.6f}")
    board.scalar("loss", i, loss)
    if i % 512 == 511:
      print("\nsaving...")
      save(vae, save_path)
      print("saved.\n")
    if i >= 65535: break # end training here
    #if i >= 2047: break # end training here


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



