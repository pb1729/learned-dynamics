import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

from atoms_display import launch_atom_display

from managan.config import get_predictor
from managan.utils import must_be


def make_linear(state):
  """ MUTATES state """
  _, nodes, must_be[3] = state.x.shape
  state.x[0, :, 1:] = 0.
  state.x[0, :, 0] = 0.8*torch.linspace(-nodes/2, nodes/2, nodes, device=state.x.device)


def main(args):
  # get comparison data
  predictor = get_predictor(args.predictor_spec, override_base=args.override)
  #assert space_dim(predictor) == 3 # TODO: salvage this somehow?
  box = None
  if args.wrap:
    box = predictor.get_box()
  if box is not None: box = np.array(box)
  def clean_for_display(state):
    if len(state.size) == 2: # handle the case where state consists of multiple time-lagged configurations
      state = state[-1] # use most recent configuration
    x = state.x_npy
    if box is not None:
      x = (x + 0.5*box) % box - 0.5*box
    return x[0]
  state = predictor.sample_q(1)
  if args.startlinear:
    make_linear(state)
  atomic_nums = state.metadata.atomic_nums if state.metadata is not None else 5*np.ones(poly_len(predictor), dtype=int)
  display = launch_atom_display(atomic_nums, clean_for_display(state))
  i = 0
  while True:
    if args.saveframes_dir is not None:
      np_img = display.get_current_screen()
      image = Image.fromarray(np_img, "RGB")
      image.save(f"{args.saveframes_dir}/{i}.png")
    if args.printevery is not None and i % args.printevery == 0:
      print(i)
    i += 1
    predictor.predict(1, state, ret=False)
    if args.slideshow:
      input("...")
    display.update_pos(clean_for_display(state))
    if args.center:
      display.center_pos()
    if args.framedelay > 0:
      time.sleep(args.framedelay)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test animation")
  parser.add_argument("predictor_spec")
  parser.add_argument("--slideshow", dest="slideshow", action="store_true")
  parser.add_argument("--framedelay", dest="framedelay", type=float, default=0.001) # default is small, but long enough so the display can update
  parser.add_argument("--center", dest="center", action="store_true")
  parser.add_argument("--wrap", dest="wrap", action="store_true")
  parser.add_argument("--startlinear", dest="startlinear", action="store_true")
  parser.add_argument("--override", dest="override", type=str, default=None)
  parser.add_argument("--printevery", dest="printevery", type=int, default=None)
  parser.add_argument("--saveframes_dir", dest="saveframes_dir", type=str, default=None)
  main(parser.parse_args())
