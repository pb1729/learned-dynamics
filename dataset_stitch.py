import argparse
from os import path, listdir, makedirs
import torch

from managan.config import get_predictor
from managan.predictor import ModelState
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
  # stitch trajectories
  file_list = [fnm for fnm in listdir(args.src_folder) if fnm[-4:] == ".bin"]
  runs = {}
  for fnm in file_list:
    *prefix, chunk_src = fnm[:-4].split("_")
    prefix = "_".join(prefix)
    chunk_src = int(chunk_src)
    if prefix not in runs:
      runs[prefix] = {}
    runs[prefix][chunk_src] = fnm
  for prefix in runs:
    trajs = []
    metadata = None
    shape = None
    for i in range(max(runs[prefix]) + 1):
      with open(path.join(args.src_folder, f"{prefix}_{i}.bin"), "rb") as f_r:
        state = read_state_from_file(f_r)
        trajs.append(state.x[::args.stride])
        if metadata is None:
          metadata = state.metadata
          shape = state.shape
    stitched_state = ModelState(shape, torch.cat(trajs, dim=0), metadata=metadata)
    with open(path.join(args.dst_folder, f"{prefix}_0.bin"), "wb") as f_w:
      save_state_to_file(f_w, stitched_state)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="dataset_chop")
  parser.add_argument("src_folder", type=str)
  parser.add_argument("dst_folder", type=str)
  parser.add_argument("--stride", type=int, default=1)
  main(parser.parse_args())
