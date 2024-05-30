import torch
import torch.nn as nn
import torch_geometric
import e3nn
from e3nn.nn.models.gate_points_2101 import Network



# This example from https://dmol.pub/applied/e3nn_traj.html




model_kwargs = {
    "irreps_in": None,  # no input features
    "irreps_hidden": e3nn.o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o"),  # hyperparameter
    "irreps_out": "1x1o",  # 12 vectors out, but only 1 vector out per input
    "irreps_node_attr": None,
    "irreps_edge_attr": e3nn.o3.Irreps.spherical_harmonics(3),
    "layers": 3,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

model = e3nn.nn.models.gate_points_2101.Network(
    **model_kwargs
)  # initializing model with parameters above

class Meanpred:
  def __init__(self, model, config):
    self.model = model
    self.config = config
    self.init_optim()
    assert config.sim.poly_len == 12
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim = torch.optim.Adam(self.model.parameters(), self.config["lr"], betas)
  @staticmethod
  def load_from_dict(states, config):
    model = e3nn.nn.models.gate_points_2101.Network(**model_kwargs).to(config.device)
    model.load_state_dict(states["model"])
    return Meanpred(model, config)
  @staticmethod
  def makenew(config):
    model = e3nn.nn.models.gate_points_2101.Network(**model_kwargs).to(config.device)
    return Meanpred(model, config)
  def save_to_dict(self):
    return {
        "model": self.model.state_dict(),
      }
  def train_step(self, data, cond):
    batch = data.shape[0]
    self.optim.zero_grad()
    loss = 0.
    for i in range(batch): # manual for loop since the network appears not to support batched data
      data_i, cond_i = data[i].reshape(self.config.sim.poly_len, 3), cond[i].reshape(self.config.sim.poly_len, 3)
      delta = data_i - cond_i
      tg_cond = torch_geometric.data.Data(x=None, pos=cond_i, y=None)
      loss = loss + ((self.model(tg_cond) - delta)**2).sum()
    loss = loss / batch # take the mean
    loss.backward()
    self.optim.step()
    return loss.item()
  def predict(self, cond):
    with torch.no_grad():
      cond = cond.reshape(self.config.sim.poly_len, 3)
      tg_cond = torch_geometric.data.Data(x=None, pos=cond, y=None)
      pred_delta = self.model(tg_cond)
      return (cond + pred_delta).reshape(1, 36)
  def set_eval(self, *args):
    pass


class MeanpredTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    N, L, state_dim = trajs.shape
    cond = self.model.config.cond(trajs[:, :-1].reshape(N*(L - 1), state_dim))
    data = trajs[:, 1:].reshape(N*(L - 1), state_dim)
    loss = self.model.train_step(data, cond)
    print(f"{i}\t ℒ = {loss:05.6f}")
    self.board.scalar("loss", i, loss)


# export model class and trainer class:
modelclass   = Meanpred
trainerclass = MeanpredTrainer



