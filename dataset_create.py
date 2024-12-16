import argparse
from os import path

from config import get_predictor
from statefiles import save_state_to_file


def main(args):
  predictor = get_predictor(args.predictor)
  for i in range(args.loop):
    state = predictor.sample_q(args.batch)
    for j in range(args.chunks):
      print(i, j)
      traj = predictor.predict(args.chunklen, state)
      filename = f"data_{args.sequence_num}_run_{i}_chunk_{j}.bin"
      with open(path.join(args.folder, filename), "wb") as f:
        save_state_to_file(f, traj)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="dataset_create")
  parser.add_argument("sequence_num", type=int)
  parser.add_argument("predictor", type=str)
  parser.add_argument("folder", type=str)
  parser.add_argument("--loop", type=int, default=128)
  parser.add_argument("--chunks", type=int, default=20)
  parser.add_argument("--chunklen", type=int, default=64)
  parser.add_argument("--batch", type=int, default=1)
  main(parser.parse_args())
