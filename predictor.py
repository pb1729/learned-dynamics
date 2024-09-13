import torch

from utils import batched_model_eval


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
    def batch(self):
      assert False, "this class not concretely implemented!"
    def __getitem__(self, key):
      assert False, "this class not concretely implemented!"
    def expand(self, key):
      assert False, "this class not concretely implemented!"
  @property
  def shape(self):
    assert False, "this class not concretely implemented!"
  def sample_q(self, batch):
    """ Some predictors may provide a way to sample States from some distribution q. (not necessarily
        the equilibrium distribution.) If so, you can sample a State of size batch by calling this function. """
    assert False, "q sampling not implemented for this class"
  def predict(self, L, state, ret=True):
    """ predict L steps based on initial state.
        MUTATES state
        ret: return the trajectory? """
    assert False, "this class not concretely implemented!"
  @staticmethod
  def from_sim(sim_nm):
    return SimPredictor(sim_nm)


class SimPredictor(Predictor):
  class State(Predictor.State):
    def __init__(self, x, v):
      self.x = x
      self.v = v
      assert x.shape == v.shape
    @property
    def batch(self):
      return self.x.shape[0]
    def __getitem__(self, key):
      return SimPredictor.State(self.x[key], self.v[key])
    def expand(self, n):
      return SimPredictor.State(
        self.x[None].expand(n, *[-1]*self.x.dim()).reshape(n*self.batch, *self.x.shape[1:]).clone(),
        self.v[None].expand(n, *[-1]*self.x.dim()).reshape(n*self.batch, *self.x.shape[1:]).clone())
    def to_model_predictor_state(self):
      return ModelPredictor.State(self.x.clone()) # model predictors only use x coordinate
  def __init__(self, sim, t_eql=4):
    self.sim = sim
    self.t_eql = t_eql
  @property
  def shape(self):
    return self.sim.shape
  def predict(self, L, state, ret=True):
    return self.sim.generate_trajectory(state.x, state.v, L, ret=ret)
  def sample_q(self, batch):
    x, v = self.sim.sample_equilibrium(batch, self.t_eql)
    return SimPredictor.State(x, v)


class ModelPredictor(Predictor):
  class State(Predictor.State):
    def __init__(self, x):
      self.x = x
    @property
    def batch(self):
      return self.x.shape[0]
    def __getitem__(self, key):
      return ModelPredictor.State(self.x[key])
    def expand(self, n):
      return ModelPredictor.State(
        self.x[None].expand(n, *[-1]*self.x.dim()).reshape(n*self.batch, *self.x.shape[1:]).clone())
    def to_model_predictor_state(self):
      return ModelPredictor.State(self.x.clone())
  def __init__(self, model):
    model.set_eval(True)
    self.model = model
  @property
  def shape(self):
    return self.model.config.predictor.shape
  def predict(self, L, state, ret=True):
    if ret:
      trajectory = torch.zeros((state.batch, L, *self.shape) ,
        dtype=state.x.dtype, device=state.x.device)
    with torch.no_grad():
      for i in range(L):
        new_x = batched_model_eval(
          (lambda x: self.model.predict(x)),
          state.x, batch=EVAL_BATCHSZ)
        if ret: trajectory[:, i] = new_x
        state.x[:] = new_x
    if ret:
      return trajectory
  def sample_q(self, batch):
    return self.get_base_predictor().sample_q(batch).to_model_predictor_state()
  def get_base_predictor(self):
    return self.model.config.predictor



