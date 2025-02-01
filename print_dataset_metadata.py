from os import path

from managan.statefiles import read_predictor_params_from_file


def main(args):
  with open(path.join(args.dataset_dir, "predictor_params.pickle"), "rb") as f:
    print(read_predictor_params_from_file(f))


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="print_model_metadata")
  parser.add_argument("dataset_dir")
  main(parser.parse_args())
