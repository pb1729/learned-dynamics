import torch
import numpy as np
from typing import Optional
from typing_extensions import Self, List, Tuple

from .utils import must_be, prod, batched_model_eval
from .sim_utils import OpenMMMetadata, OpenMMSimError
from .sims import sims



def assert_shapes_compatible(base_shape, shape):
  assert len(base_shape) == len(shape), "number of dimensions does not match"
  for d_base, d in zip(base_shape, shape):
    assert d_base == d or d_base == -1, f"dimension mismatch {d_base} != {d}"


class State:
  """ Base State class. This one pretty much does nothing, but its subclasses do stuff.
      State can have multiplicity, and the amount of multiplicity is defined by the batch property. """
  def __init__(self, size:tuple, shape:tuple):
    self._size = size    # eg. (batch) or (time, batch)
    self._shape = shape  # eg. (natoms, 3)
  @property
  def size(self):
    return self._size
  @property
  def shape(self):
    return self._shape
  @staticmethod
  def _check_size_shape(tens, base_shape, base_size=None):
    ndims = len(base_shape)
    size = tens.shape[:-ndims]
    if base_size is not None:
      assert size == base_size, f"size mismatch {size} != {base_size}"
    shape = tens.shape[-ndims:]
    assert_shapes_compatible(base_shape, shape)
    return size, shape
  @property
  def x(self) -> torch.Tensor:
    assert False, "this class not concretely implemented"
  @property
  def x_npy(self) -> np.ndarray:
    return self.x.cpu().numpy()
  def __getitem__(self, key) -> Self:
    assert False, "this class not concretely implemented!"
  def expand(self, n, dim) -> Self:
    assert False, "expand not implemented for this State subclass"
  def _get_expand_shape(self, n, dim):
    assert dim <= len(self.size), f"can't unsqueeze dim {dim} for State of shape {self.size}"
    ans = self.size + self.shape
    ans = ans[:dim] + (n,) + ans[dim:]
    return ans
  def reshape(self, *newsize) -> Self:
    assert False, "reshape not implemented for this State subclass"
  def _reshape_tens(self, tens, newsize):
    assert prod(newsize) == prod(self.size), f"can't reshape {self.size} to {newsize}"
    return tens.reshape(*(newsize + self.shape))
  @property
  def metadata(self) -> Optional[OpenMMMetadata]:
    return None
class Predictor:
  """ Base class for all Predictors. Predictors have 1 basic function: they simulate
      stochastic dynamics. They can be passed an instance of their State class and can
      MUTATE it over time according to the dynamics. They can also produce *trajectories*.
      Note that a trajectory is not necessarily a sequence of State's. For example, a sim
      has state consisting of position and velocity, but its trajectory is only a sequence
      of positions. Though state can be very complex in terms of the data it stores, a
      trajectory is just a tensor of shape (batch, L, *self.shape). """
  def shape(self) -> tuple:
    assert False, "this class not concretely implemented!"
  def sample_q(self, batch:int) -> State:
    """ Some predictors may provide a way to sample States from some distribution q. (not necessarily
        the equilibrium distribution.) If so, you can sample a State of size batch by calling this function. """
    assert False, "q sampling not implemented for this class"
  def predict(self, L:int, state:State, ret:bool=True) -> Optional[State]:
    """ predict L steps based on initial state.
        MUTATES state
        ret: return the trajectory as a ModelState """
    assert False, "this class not concretely implemented!"
  def get_box(self) -> Optional[Tuple[float, float, float]]:
    return None


class ModelState(State):
  def __init__(self, shape, x, **kwargs):
    size, shape = self._check_size_shape(x, shape)
    self._x = x
    self.kwargs = kwargs
    super().__init__(size, shape)
  @property
  def x(self) -> torch.Tensor:
    return self._x
  def __getitem__(self, key):
    return ModelState(self.shape, self._x[key], **self.kwargs)
  def expand(self, n, dim):
    expand_shape = self._get_expand_shape(n, dim)
    x_new = self._x.unsqueeze(dim)
    return ModelState(self.shape, x_new.expand(*expand_shape).clone(), **self.kwargs)
  def reshape(self, *newsize):
    return ModelState(self.shape, self._reshape_tens(self._x, newsize), **self.kwargs)
  def to_model_predictor_state(self):
    return ModelState(self.shape, self._x.clone(), **self.kwargs)
  @property
  def metadata(self):
    if "metadata" in self.kwargs:
      return self.kwargs["metadata"]
    else:
      return None
class ModelPredictor(Predictor):
  def __init__(self, model):
    if hasattr(model, "set_eval"):
      model.set_eval(True)
    self.model = model
  def shape(self):
    return self.model.config.predictor.shape()
  def predict(self, L, state, ret=True):
    """ MUTATES state """
    if ret:
      trajectory = torch.zeros(L, *state.size, *state.shape,
        dtype=state._x.dtype, device=state._x.device)
    with torch.no_grad():
      for i in range(L):
        # TODO: do we still need to use batched_model_eval here?
        new_x = self.model.predict(state)
        state._x[:] = new_x
        if ret: trajectory[i] = new_x
    if ret:
      return ModelState(state.shape, trajectory, **state.kwargs)
  def sample_q(self, batch):
    return self.get_base_predictor().sample_q(batch).to_model_predictor_state()
  def get_base_predictor(self) -> Predictor:
    return self.model.config.predictor
  def get_box(self):
    return self.get_base_predictor().get_box()


try:
  from .openmm_sims import OpenMMConfig, OpenMMMetadata, XReporter, openmm_sims
  from openmm import OpenMMException
except ModuleNotFoundError:
  print("Warning: import of openmm failed, skipping.")
  def get_openmm_predictor(key):
    assert False, "openmm failed to import, so construction of predictor is impossible"
else:
  class OpenMMState(State):
    def __init__(self, metadata:OpenMMMetadata, sims:List, reporters:List[XReporter]):
      self._metadata = metadata
      self.sims = sims
      self.reporters = reporters
      size = (len(sims),)
      shape = (metadata.atomic_nums.shape[0], 3)
      super().__init__(size, shape)
    @property
    def x(self) -> torch.Tensor:
      return torch.tensor(self.x_npy, dtype=torch.float32, device="cuda")
    @property
    def x_npy(self) -> np.ndarray:
      return np.array([self.reporter_x(reporter) for reporter in self.reporters])
    def reporter_x(self, reporter) -> np.ndarray:
      x = reporter.x
      assert x is not None, "trying to read uninitialized reporter"
      return x[self._metadata.atom_indices]
    def __getitem__(self, key) -> Self:
      return OpenMMState(self._metadata, self.sims[key], self.reporters[key])
    def to_model_predictor_state(self):
      return ModelState(self.shape, self.x, metadata=self._metadata)
    @property
    def metadata(self) -> Optional[OpenMMMetadata]:
      return self._metadata
  class OpenMMPredictor(Predictor):
    def __init__(self, openmm_config:OpenMMConfig):
      self.openmm_config = openmm_config
    def shape(self):
      return (-1, 3) # number of atoms can vary
    def sample_q(self, batch) -> State:
      metadata, sims, reporters = self.openmm_config.sample_q(batch)
      return OpenMMState(metadata, sims, reporters)
    def predict(self, L, state, ret=True):
      """ MUTATES state """
      if ret:
        trajectory = torch.zeros(L, *state.size, *state.shape, dtype=torch.float32, device="cuda")
      try:
        for i, sim in enumerate(state.sims):
          for j in range(L):
            sim.step(self.openmm_config.dt)
            if ret:
              trajectory[j, i] = torch.tensor(state.reporter_x(state.reporters[i]), dtype=torch.float32, device="cuda")
      except OpenMMException as e:
        raise OpenMMSimError(e)
      if ret:
        return ModelState(state.shape, trajectory, metadata=state.metadata)
    def get_box(self):
      boxsz = self.openmm_config.boxsz
      return (boxsz, boxsz, boxsz)
  def get_openmm_predictor(key):
    return OpenMMPredictor(openmm_sims[key])
