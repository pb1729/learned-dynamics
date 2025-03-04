import argparse
from os import path, listdir, makedirs

from managan.config import get_predictor
from managan.statefiles import save_state_to_file, read_state_from_file


def main(args):
  # Ensure destination folder doesn't exist yet
  if path.exists(args.dst_folder):
    assert False, f"Destination folder {args.dst_folder} already exists"
  else:
    makedirs(args.dst_folder)
  # copy over predictor params
  with open(path.join(args.src_folder, "predictor_params.pickle"), "rb") as f_r:
    with open(path.join(args.dst_folder, "predictor_params.pickle"), "wb") as f_w:
      f_w.write(f_r.read())
  # chop trajectories
  file_list = [fnm for fnm in listdir(args.src_folder) if fnm[-4:] == ".bin"]
  for fnm in file_list:
    with open(path.join(args.src_folder, fnm), "rb") as f_r:
      state = read_state_from_file(f_r)
      *prefix, chunk_src = fnm[:-4].split("_")
      prefix = "_".join(prefix)
      chunk_src = int(chunk_src)
      L, batch = state.size
      mult = L // args.chunklen
      for i, base in enumerate(range(L - mult*args.chunklen, L, args.chunklen)): # start as late as possible
        segment = state[base:base + args.chunklen]
        chunk_dst = mult*chunk_src + i
        with open(path.join(args.dst_folder, f"{prefix}_{chunk_dst}.bin"), "wb") as f_w:
          save_state_to_file(f_w, segment)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="dataset_chop")
  parser.add_argument("src_folder", type=str)
  parser.add_argument("dst_folder", type=str)
  parser.add_argument("--chunklen", type=int, default=64)
  main(parser.parse_args())
