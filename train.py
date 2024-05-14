import itertools

from run_visualization import TensorBoard
from sims import dataset_gen
from config import Config, load, save, makenew



def train(model, save_path):
  assert save_path.split(".")[-1] == "pt", "expected pytorch .pt file suffix"
  run_name = ".".join(save_path.split("/")[-1].split(".")[:-1])
  print(run_name)
  print(model.config)
  board = TensorBoard(run_name)
  config = model.config # configuration for this run...
  data_generator = dataset_gen(config.sim, config.batch, config.simlen,
    t_eql=config.t_eql, subtract_cm=config.subtract_mean, x_only=config.x_only)
  trainer = config.trainerclass(model, board)
  for i in itertools.count():
    trajs = data_generator.send(None if i < config.nsteps else True)
    if trajs is None: break
    trainer.step(i, trajs) # main training step
    if (i + 1) % config.save_every == 0:
      print("\nsaving...")
      save(model, save_path)
      print("saved.\n")


def training_run(save_path, src):
  if isinstance(src, Config): # create new from config
    model = makenew(src)
  elif isinstance(src, str): # load from path
    model = load(src)
  else:
    raise TypeError("incorrect source for training run!")
  train(model, save_path)



