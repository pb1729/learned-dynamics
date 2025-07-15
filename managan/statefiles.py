from io import BufferedReader, BufferedWriter, BytesIO
from struct import pack, unpack
from os import listdir, path
import pickle
import numpy as np
import torch
import random

from .predictor import State, Predictor, ModelState


class DatasetError(Exception):
  pass


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

def save_predictor_params_to_file(file:BufferedWriter, predictor):
  box = predictor.get_box()
  pickle.dump({
    "box": box,
    "shape": predictor.shape(),
    "name": predictor.name,
  }, file)

def read_predictor_params_from_file(file:BufferedReader):
  metadata = pickle.load(file)
  return metadata


class DatasetState(State):
  def __init__(self, fnm, batch=1):
    with open(fnm, "rb") as f:
      self.traj = read_state_from_file(f)
    self.L, saved_batch, *rest = self.traj.size
    self._size = (batch,) + tuple(rest)
    self._shape = self.traj.shape
    assert batch <= saved_batch, f"requested batch {batch} cannot exceed saved batch {saved_batch}"
    self.batch = batch
    self.t = 0
  @property
  def x(self) -> torch.Tensor:
    return self.traj[self.t, :self.batch].x
  @property
  def metadata(self):
    return self.traj.metadata
  def to_model_predictor_state(self):
    return ModelState(self.shape, self.x.clone(), metadata=self.metadata)

class DatasetPredictor(Predictor):
  def __init__(self, dataset_dir, stride=1, repeat:bool=False):
    self.dataset_dir = dataset_dir
    self.stride = stride
    self.repeat = repeat # if True, allows repeating the dataset across multiple epochs
    with open(path.join(dataset_dir, "predictor_params.pickle"), "rb") as f:
      params = read_predictor_params_from_file(f)
    self._box = params["box"]
    self._shape = params["shape"]
    self.file_list = [fnm for fnm in listdir(dataset_dir) if fnm[-4:] == ".bin"]
    random.shuffle(self.file_list)
    self.file_index = 0
  def get_box(self):
    return self._box
  def shape(self):
    return self._shape
  def sample_q(self, batch):
    if not self.file_index < len(self.file_list):
      if self.repeat:
        print("EPOCH++")
        random.shuffle(self.file_list)
        self.file_index = 0
      else:
        raise DatasetError("out of files in dataset!")
    ans = DatasetState(path.join(self.dataset_dir, self.file_list[self.file_index]), batch)
    self.file_index += 1
    return ans
  def predict(self, L, state, ret=True):
    assert isinstance(state, DatasetState), "DatasetPredictor can only handle DatasetState's"
    assert L*self.stride < state.L - state.t, f"DatasetPredictor tried to predict {self.stride}*{L} steps, but only {state.L - state.t - 1} data remains"
    t_old = state.t
    state.t += L*self.stride
    if ret:
      return state.traj[t_old + self.stride : state.t + 1 : self.stride, :state.batch]

def get_dataset_predictor(spec):
  return DatasetPredictor(spec)

def get_strided_dataset_predictor(spec):
  stride, dataset_dir = spec.split("?")
  stride = int(stride)
  return DatasetPredictor(dataset_dir, stride=stride)

def get_repeated_dataset_predictor(spec):
  stride, dataset_dir = spec.split("?")
  stride = int(stride)
  return DatasetPredictor(dataset_dir, stride=stride, repeat=True)
