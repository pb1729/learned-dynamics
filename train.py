import itertools
from threading import Thread
from queue import Queue
import os

import torch

from run_visualization import TensorBoard
from config import Config, load, save, makenew


def dataset_gen(config):
  """ generate many datasets in a separate thread
      one should use the send() method for controlling this generator, calling
      send(True) if more data will be required and send(False) otherwise """
  data_queue = Queue(maxsize=8) # we set a maxsize to control the number of items taking up memory on GPU
  control_queue = Queue()
  def thread_main():
    while True: # queue maxsize stops us from going crazy here
      state = config.predictor.sample_q(16*config.batch)
      next_dataset = config.predictor.predict(config.simlen, state)
      for i in range(0, 16*config.batch, config.batch):
        if not control_queue.empty():
          command = control_queue.get_nowait()
          if command == "halt":
            return
        data_queue.put(next_dataset[i:i+config.batch])
  t = Thread(target=thread_main)
  t.start()
  while True:
    data = data_queue.get()
    halt = yield data # "keep going" is encoded as None, since python requires the first send() to be passed a None anyway
    if halt is not None:
      control_queue.put("halt")
      yield None
      break


def train(model, save_path):
  assert save_path.split(".")[-1] == "pt", "expected pytorch .pt file suffix"
  if os.path.isfile(save_path):
    input(f"File {save_path} already exists. Press ENTER to continue training anyway. Press CTL-C to exit.")
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  print(model.config)
  board = TensorBoard(run_name)
  config = model.config # configuration for this run...
  data_generator = dataset_gen(config)
  trainer = config.trainerclass(model, board)
  if isinstance(config.nsteps, list):
    nsteps = max(config.nsteps)
    checkpoints = [ns for ns in config.nsteps if ns < nsteps]
  else:
    nsteps = config.nsteps
    checkpoints = []
  for i in itertools.count():
    trajs = data_generator.send(None if i < nsteps else True)
    if trajs is None: break
    trainer.step(i, trajs) # main training step
    if (i + 1) % config.save_every == 0:
      print("\nsaving...")
      save(model, save_path)
      if i + 1 in checkpoints:
        save(model, save_path[:-3] + ".chkp_" + str(i + 1) + ".pt")
      # checkpoint
      print("saved.\n")


def training_run(save_path, src):
  if isinstance(src, Config): # create new from config
    model = makenew(src)
  elif isinstance(src, str): # load from path
    model = load(src)
  else:
    raise TypeError("incorrect source for training run!")
  train(model, save_path)
