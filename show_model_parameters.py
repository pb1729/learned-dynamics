from managan.config import load
from managan.utils import prod


def main(args):
  model = load(args.modelpath)
  net = getattr(model, args.netnm)
  n_params = 0
  for name, param in net.named_parameters():
    param_sz = prod(param.shape)
    if args.verbose:
      print(name, param_sz)
    n_params += prod(param.shape)
  print(f"Total Parameters: {n_params}")


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="print_model_metadata")
  parser.add_argument("modelpath")
  parser.add_argument("--netnm", type=str, default="dn")
  parser.add_argument("--verbose", action="store_true")
  main(parser.parse_args())



