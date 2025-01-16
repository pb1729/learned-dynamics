from managan.config import load_config, load, save
from managan.predictor import Predictor


def main(args):
  try:
    config = load_config(args.modelpath)
  except FileNotFoundError:
    config = load_config(args.modelpath, override_base="dummy:original predictor could not be loaded")
  print(config)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="print_model_metadata")
  parser.add_argument("modelpath")
  main(parser.parse_args())
