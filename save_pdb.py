from managan.config import get_predictor
from managan.backmap_to_pdb import model_state_to_pdb


def main(args):
  predictor = get_predictor(f"dataset:{args.dataset_path}")
  state = predictor.sample_q(1)
  predictor.predict(args.idx, state, ret=False)
  pdb = model_state_to_pdb(state.to_model_predictor_state()[0])
  with open(args.save_path, "w") as f:
    f.write(pdb)
  print("saved.")


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="save_pdb")
  parser.add_argument("dataset_path", type=str)
  parser.add_argument("idx", type=int)
  parser.add_argument("save_path", type=str)
  main(parser.parse_args())
