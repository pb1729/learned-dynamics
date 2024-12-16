from io import BufferedReader, BufferedWriter, BytesIO
from struct import pack, unpack
import pickle
import numpy as np
import torch

from predictor import ModelState
from config import get_predictor


def save_state_to_file(file:BufferedWriter, state:ModelState):
  # deconstruct state
  metadata = state.metadata
  shape = state.shape
  x_bytes = BytesIO()
  np.save(x_bytes, state.x_npy, allow_pickle=False)
  subfiles = [
    pickle.dumps(metadata),
    pickle.dumps(shape),
    x_bytes.getvalue()]
  # store subfiles
  lengths = [len(subfile) for subfile in subfiles]
  file.write(pack("!III", *lengths))
  for subfile in subfiles:
    file.write(subfile)

def read_state_from_file(file:BufferedReader):
  # read subfiles
  lengths = unpack("!III", file.read(3*4))
  subfiles = []
  for length in lengths:
    subfiles.append(file.read(length))
  # reconstruct state
  metadata = pickle.loads(subfiles[0])
  shape = pickle.loads(subfiles[1])
  x_npy = np.load(BytesIO(subfiles[2]), allow_pickle=False)
  return ModelState(shape, torch.tensor(x_npy, device="cuda"), metadata=metadata)

if __name__ == "__main__":
  predictor = get_predictor("openmm:SEQ_t10_L20_seqAAAA")
  state = predictor.sample_q(1)
  buffer = BytesIO()
  save_state_to_file(buffer, predictor.predict(3, state))
  buffer.seek(0) # reset position as though we were just reopening the file now
  read_state_from_file(buffer)
