import torch
import torch.nn as nn
from torchview import draw_graph

from managan.config import get_predictor
from managan.run_visualization import DummyTensorBoard


def hunt_modules(model):
  """ Hunts for attributes of the model that are nn.Module's or lists thereof. This is
      pure jank and you should NEVER feel bad about making changes that break it. """
  ans = []
  for attr in vars(model).values():
    if isinstance(attr, nn.Module):
      ans.append(attr)
    elif isinstance(attr, list):
      for item in attr:
        if isinstance(item, nn.Module):
          ans.append(item)
  return ans


def main(args):
  # load model
  predictor:ModelPredictor = get_predictor("model:" + args.fpath, override_base=args.override)
  # setup to make sure we're profiling a full training step
  model = predictor.model
  config = model.config
  trainer = config.trainerclass(model, DummyTensorBoard("draw graph test"))
  model.set_eval(False)
  modules = hunt_modules(model)
  inputs = [None for module in modules]
  # setup input recording
  for i, mod in enumerate(modules):
    mod.loop_index = i
    def record_input_fwd_pre_hook(module, inp):
      inputs[module.loop_index] = inp
    mod.register_forward_pre_hook(record_input_fwd_pre_hook)
  # record input
  state = predictor.sample_q(1)
  traj = predictor.predict(args.length, state)
  try:
    trainer.step(0, traj)
  except RuntimeError as e:
    print(e)
    print("'RuntimeError: CUDA error: invalid configuration argument' means you probably set length too low and are trying to do ops on zero size tensors")
    quit()
  # record graphs
  for i, (inp, mod) in enumerate(zip(inputs, modules)):
    module_graph = draw_graph(mod, inp, expand_nested=True, depth=args.depth)
    module_graph.visual_graph.render("images/%s_modelno_%d" % (args.fpath.split("/")[-1], i), format="svg")
  print("done. saved into images/ dir")


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="draw_graph")
  parser.add_argument("fpath")
  parser.add_argument("--length", dest="length", type=int, default=8)
  parser.add_argument("--depth", dest="depth", type=int, default=2)
  parser.add_argument("--override")
  main(parser.parse_args())



