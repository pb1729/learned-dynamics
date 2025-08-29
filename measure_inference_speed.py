import time
import torch

from managan.backmap_to_pdb import model_states_to_mdtrajs
from managan.utils import must_be
from plotting.predictor_argparse_util import args_to_predictor_list, add_model_list_arg


STEPS = 64          # how many inference steps to time
WARMUP_STEPS = 10   # used for burn-in


def main(args):
  predictors = args_to_predictor_list(args)
  angles = {}
  for predictor in predictors:
    print(f"Testing {predictor.name}")
    state = predictor.sample_q(args.batch)
    print("Burning in predictor...")
    predictor.predict(WARMUP_STEPS, state)
    print("Done. Measuring inference speed...")
    torch.cuda.synchronize()     # make sure the GPU is idle
    t0 = time.perf_counter()
    predictor.predict(STEPS, state)
    torch.cuda.synchronize()     # wait for kernels to finish
    elapsed = time.perf_counter() - t0
    print(f"{STEPS} in {elapsed}s")
    print(f"{STEPS/elapsed} steps/wall-s")



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--batch", type=int, default=1, help="batch size")
  main(parser.parse_args())
