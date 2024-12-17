import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from atoms_display import launch_atom_display

from polymer_util import poly_len, space_dim
from config import get_predictor
from utils import must_be


SHOW_REALSPACE_SAMPLES = 10
RADIAL = True


def compare_predictions_x(x_init, x_actual, predictor, basis, show_histogram=True):
  x_init   = x_init  .cpu().numpy().reshape(-1, poly_len(predictor), space_dim(predictor))
  x_actual = x_actual.cpu().numpy().reshape(-1, poly_len(predictor), space_dim(predictor))
  plotter = Plotter(BASES[basis], samples_subset_size=SHOW_REALSPACE_SAMPLES, title=("basis type = " + basis))
  plotter.plot_samples(x_actual)
  plotter.plot_samples_ic(x_init)
  plotter.show()
  if show_histogram:
    if RADIAL:
      plotter.plot_hist_radial(x_actual)
    else:
      plotter.plot_hist(x_actual)
      plotter.plot_hist_ic(x_init)
    plotter.show()


def eval_sample_step(init_states, fin_statess, predictor, basis):
  batch, contins, _ = fin_statess.shape
  for i in range(batch):
    print("\nnext i.c: i=%d\n" % i)
    init_state = init_states[i, None]
    fin_states = fin_statess[i, :]
    compare_predictions_x(init_state, fin_states, predictor, basis)


def make_linear(state):
  """ MUTATES state """
  _, nodes, must_be[3] = state.x.shape
  state.x[0, :, 1:] = 0.
  state.x[0, :, 0] = 0.8*torch.linspace(-nodes/2, nodes/2, nodes, device=state.x.device)


def main(args):
  # get comparison data
  predictor = get_predictor(args.predictor_spec)
  assert space_dim(predictor) == 3
  box = None
  if args.wrap:
    box = predictor.get_box()
  if box is not None: box = box.cpu().numpy()
  def clean_for_display(x):
    if box is not None:
      x = (x + 0.5*box) % box - 0.5*box
    return x[0]
  if args.override is not None:
    override_predictor = get_predictor(args.override)
    override_state = override_predictor.sample_q(1)
    # actual state will be a ModelState (overrides only supported if main predictor is a ModelPredictor)
    state = override_predictor.predict(1, override_state)[0]
  else:
    state = predictor.sample_q(1)
  if args.startlinear:
    make_linear(state)
  atomic_nums = state.metadata.atomic_nums if state.metadata is not None else 5*np.ones(poly_len(predictor), dtype=int)
  display = launch_atom_display(atomic_nums, clean_for_display(state.x_npy))
  while True:
    predictor.predict(1, state, ret=False)
    if args.slideshow:
      input("...")
    display.update_pos(clean_for_display(state.x_npy))
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
  main(parser.parse_args())
