import torch

from managan.config import get_predictor
from managan.utils import must_be, avg_relative_diff
from managan.predictor import ModelState
from managan.symmetries import RotSymm, TransSymm, check_symm


def main(args):
  print("Test entire model from file.")
  predictor = get_predictor("model:" + args.modelfile, override_base=args.override)
  if predictor.get_box() is not None:
    print("Warning, some inaccuracy could arise due to self-interactions through the periodic box.")
  model = predictor.model
  state = predictor.sample_q(1)
  x = state.x.clone()
  for symm in [RotSymm(), TransSymm()]:
    def predict(x):
      # fix a random seed
      model.randgen.set_transform(None, seed=0x947)
      model_state = ModelState(state.shape, x, **state.kwargs)
      return model.predict(model_state)
    check_symm(symm, predict, [x], ["p"], ["p"])




if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="symmetry_checker")
  parser.add_argument("modelfile", type=str)
  parser.add_argument("--override", type=str, default=None)
  main(parser.parse_args())
