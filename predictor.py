import torch
import numpy as np
from typing_extensions import Self, List

from utils import batched_model_eval
from hoomd_sims import HoomdSim
from openmm_sims import OpenMMConfig, OpenMMMetadata, XReporter


EVAL_BATCHSZ = 1024


class Predictor:
  """ Base class for all Predictors. Predictors have 1 basic function: they simulate
      stochastic dynamics. They can be passed an instance of their State class and can
      MUTATE it over time according to the dynamics. They can also produce *trajectories*.
      Note that a trajectory is not necessarily a sequence of State's. For example, a sim
      has state consisting of position and velocity, but its trajectory is only a sequence
      of positions. Though state can be very complex in terms of the data it stores, a
      trajectory is just a tensor of shape (batch, L, *self.shape). """
  class State:
    """ Base State class. This one pretty much does nothing, but its subclasses do stuff.
        State can have multiplicity, and the amount of multiplicity is defined by the batch property. """
    @property
    def batch(self) -> int:
      assert False, "this class not concretely implemented!"
    @property
    def atomic_nums(self) -> np.ndarray|None:
      return None
    @property
    def x(self) -> torch.Tensor:
      assert False, "this class not concretely implemented"
    @property
    def x_npy(self) -> np.ndarray:
      return self.x.cpu().numpy()
    def __getitem__(self, key) -> Self:
      assert False, "this class not concretely implemented!"
    def expand(self, n) -> Self:
      assert False, "this class not concretely implemented!"
  @property
  def shape(self) -> tuple:
    assert False, "this class not concretely implemented!"
  def sample_q(self, batch) -> State:
    """ Some predictors may provide a way to sample States from some distribution q. (not necessarily
        the equilibrium distribution.) If so, you can sample a State of size batch by calling this function. """
    assert False, "q sampling not implemented for this class"
  def predict(self, L, state, ret=True) -> torch.Tensor | None:
    """ predict L steps based on initial state.
        MUTATES state
        ret: return the trajectory? """
    assert False, "this class not concretely implemented!"
  def get_box(self) -> torch.Tensor|None:
    return None


class SimPredictor(Predictor):
  class State(Predictor.State):
    def __init__(self, x, v):
      self._x = x
      self._v = v
      assert x.shape == v.shape
    @property
    def batch(self):
      return self._x.shape[0]
    @property
    def x(self) -> torch.Tensor:
      return self._x
    def __getitem__(self, key):
      return SimPredictor.State(self._x[key], self._v[key])
    def expand(self, n):
      return SimPredictor.State(
        self._x[None].expand(n, *[-1]*self._x.dim()).reshape(n*self.batch, *self._x.shape[1:]).clone(),
        self._v[None].expand(n, *[-1]*self._x.dim()).reshape(n*self.batch, *self._x.shape[1:]).clone())
    def to_model_predictor_state(self):
      return ModelPredictor.State(self._x.clone()) # model predictors only use x coordinate
  def __init__(self, sim, t_eql=4):
    self.sim = sim
    self.t_eql = t_eql
  @property
  def shape(self):
    return self.sim.shape
  def predict(self, L, state, ret=True):
    """ MUTATES state """
    return self.sim.generate_trajectory(state.x, state.v, L, ret=ret)
  def sample_q(self, batch):
    x, v = self.sim.sample_equilibrium(batch, self.t_eql)
    return SimPredictor.State(x, v)


class ModelPredictor(Predictor):
  class State(Predictor.State):
    def __init__(self, x, **kwargs):
      self._x = x
      self.kwargs = kwargs
    @property
    def batch(self):
      return self._x.shape[0]
    @property
    def x(self) -> torch.Tensor:
      return self._x
    def __getitem__(self, key):
      return ModelPredictor.State(self._x[key])
    def expand(self, n):
      return ModelPredictor.State(
        self._x[None].expand(n, *[-1]*self._x.dim()).reshape(n*self.batch, *self._x.shape[1:]).clone())
    def to_model_predictor_state(self):
      return ModelPredictor.State(self._x.clone())
  def __init__(self, model):
    model.set_eval(True)
    self.model = model
  @property
  def shape(self):
    return self.model.config.predictor.shape
  def predict(self, L, state, ret=True):
    """ MUTATES state """
    if ret:
      trajectory = torch.zeros((state.batch, L, *self.shape) ,
        dtype=state.x.dtype, device=state.x.device)
    with torch.no_grad():
      for i in range(L):
        new_x = batched_model_eval(
          (lambda x: self.model.predict(x, **state.kwargs)),
          state.x, batch=EVAL_BATCHSZ)
        if ret: trajectory[:, i] = new_x
        state.x[:] = new_x
    if ret:
      return trajectory
  def sample_q(self, batch):
    return self.get_base_predictor().sample_q(batch).to_model_predictor_state()
  def get_base_predictor(self) -> Predictor:
    return self.model.config.predictor
  def get_box(self):
    return self.get_base_predictor().get_box()


class HoomdPredictor(Predictor):
  class State(Predictor.State):
    def __init__(self, simulations):
      self.simulations = simulations
    @staticmethod
    def sim2pos(simulation):
      snapshot = simulation.state.get_snapshot()
      return snapshot.particles.position + np.array(snapshot.configuration.box[:3])*snapshot.particles.image
    @property
    def x(self) -> torch.Tensor:
      return torch.tensor(self.x_npy, dtype=torch.float32, device="cuda")
    @property
    def x_npy(self) -> np.ndarray:
      return np.array([self.sim2pos(sim) for sim in self.simulations])
    @property
    def batch(self):
      return len(self.simulations)
    def __getitem__(self, key) -> Self:
      return HoomdPredictor.State(self.simulations[key])
    def to_model_predictor_state(self):
      return ModelPredictor.State(self.x)
  def __init__(self, hoomd_sim:HoomdSim):
    self.hoomd_sim = hoomd_sim
  @property
  def shape(self):
    return (self.hoomd_sim.n_particles, 3)
  def sample_q(self, batch) -> State:
    return HoomdPredictor.State([self.hoomd_sim.sample_q() for _ in range(batch)])
  def predict(self, L, state, ret=True):
    """ MUTATES state """
    if ret:
      trajectory = torch.zeros((state.batch, L, *self.shape), dtype=torch.float32, device="cuda")
    for i, sim in enumerate(state.simulations):
      for j in range(L): # probably slightly faster/more cache friendly to make this the inside loop
        self.hoomd_sim.step(sim)
        if ret:
          trajectory[i, j] = torch.tensor(state.sim2pos(sim), device="cuda")
    if ret:
      return trajectory
  def get_box(self):
    return torch.tensor(self.hoomd_sim.box, dtype=torch.float32, device="cuda")


class OpenMMPredictor(Predictor):
  class State(Predictor.State):
    def __init__(self, metadata:OpenMMMetadata, sims:List, reporters:List[XReporter]):
      self.metadata = metadata
      self.sims = sims
      self.reporters = reporters
    @property
    def x(self) -> torch.Tensor:
      return torch.tensor(self.x_npy, dtype=torch.float32, device="cuda")
    @property
    def x_npy(self) -> np.ndarray:
      return np.array([self.reporter_x(reporter) for reporter in self.reporters])
    def reporter_x(self, reporter) -> np.ndarray:
      x = reporter.x
      assert x is not None, "trying to read uninitialized reporter"
      return x[self.metadata.atom_indices]
    @property
    def batch(self):
      return len(self.sims)
    @property
    def atomic_nums(self) -> np.ndarray:
      return self.metadata.atomic_nums
    def __getitem__(self, key) -> Self:
      return OpenMMPredictor.State(self.metadata, self.sims[key], self.reporters[key])
    def to_model_predictor_state(self):
      return ModelPredictor.State(self.x, metadata=self.metadata)
  def __init__(self, openmm_config:OpenMMConfig):
    self.openmm_config = openmm_config
  @property
  def shape(self):
    return (-1, 3) # number of atoms can vary
  def sample_q(self, batch) -> State:
    metadata, sims, reporters = self.openmm_config.sample_q(batch)
    return OpenMMPredictor.State(metadata, sims, reporters)
  def predict(self, L, state, ret=True):
    """ MUTATES state """
    if ret:
      trajectory = torch.zeros((state.batch, L, *self.shape), dtype=torch.float32, device="cuda")
    for i, sim in enumerate(state.sims):
      for j in range(L):
        sim.step(self.openmm_config.dt)
        if ret:
          trajectory[i, j] = torch.tensor(state.reporter_x(state.reporters[i]), dtype=torch.float32, device="cuda")
    if ret:
      return trajectory
  def get_box(self):
    boxsz = self.openmm_config.boxsz
    return torch.tensor((boxsz, boxsz, boxsz), device="cuda")
