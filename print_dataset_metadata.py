from os import path

from managan.statefiles import read_predictor_params_from_file, read_state_from_file


def main(args):
  with open(path.join(args.dataset_dir, "predictor_params.pickle"), "rb") as f:
    print(read_predictor_params_from_file(f))
  if args.show_trajec_stats is not None:
    with open(path.join(args.dataset_dir, args.show_trajec_stats), "rb") as f:
      print("trajec size:", read_state_from_file(f).size)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="print_model_metadata")
  parser.add_argument("dataset_dir")
  parser.add_argument("--show_trajec_stats", type=str, default=None)
  main(parser.parse_args())
