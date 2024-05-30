import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import e3nn
from e3nn.nn.models.gate_points_2101 import Network

from gan_common import GANTrainer


# This example from https://dmol.pub/applied/e3nn_traj.html

k_L = 1.0
NOISE_IRREPS = e3nn.o3.Irreps("2x1o")

gen_kwargs = {
    "irreps_in": NOISE_IRREPS,  # random noise input
    "irreps_hidden": e3nn.o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o + 3x2e"),  # hyperparameter
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

disc_kwargs = {
    "irreps_in": e3nn.o3.Irreps("1x1o"), # take (data - cond) as input
    "irreps_hidden": e3nn.o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o + 3x2e"),  # hyperparameter
    "irreps_out": "1x0e",  # scalar output
    "irreps_node_attr": None,
    "irreps_edge_attr": e3nn.o3.Irreps.spherical_harmonics(3),
    "layers": 3,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "reduce_output": True,  # reduce output to a single scalar!
}


class GAN:
  is_gan = True
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen = gen
    self.config = config
    self.init_optim()
    assert config.sim.poly_len == 12
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.Adam(self.disc.parameters(), self.config["lr_d"], betas)
    self.optim_g = torch.optim.Adam(self.gen.parameters(), self.config["lr_g"], betas)
  @staticmethod
  def load_from_dict(states, config):
    disc = e3nn.nn.models.gate_points_2101.Network(**disc_kwargs).to(config.device)
    gen = e3nn.nn.models.gate_points_2101.Network(**gen_kwargs).to(config.device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return GAN(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc = e3nn.nn.models.gate_points_2101.Network(**disc_kwargs).to(config.device)
    gen = e3nn.nn.models.gate_points_2101.Network(**gen_kwargs).to(config.device)
    return GAN(disc, gen, config)
  def save_to_dict(self):
    return {
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    data, cond = data.reshape(-1, 12, 3), cond.reshape(-1, 12, 3)
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, data, cond):
    self.optim_d.zero_grad()
    # train on real data (with instance noise)
    instance_noise_r = self.config["inst_noise_str_r"]*torch.randn_like(data)
    r_data = data + instance_noise_r # instance noise
    y_r = self.eval_disc(cond, r_data)
    # train on generated data (with instance noise)
    g_data = self.eval_gen(cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.eval_disc(cond, g_data)
    # endpoint penalty on interpolated data
    mix_factors1 = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    mixed_data1 = mix_factors1*g_data + (1 - mix_factors1)*r_data
    y_mixed1 = self.eval_disc(cond, mixed_data1)
    mix_factors2 = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    mixed_data2 = mix_factors2*g_data + (1 - mix_factors2)*r_data
    y_mixed2 = self.eval_disc(cond, mixed_data2)
    ep_penalty = (self.endpoint_penalty(r_data, g_data, y_r, y_g)
                + self.endpoint_penalty(mixed_data1, r_data, y_mixed1, y_r)
                + self.endpoint_penalty(g_data, mixed_data1, y_g, y_mixed1)
                + self.endpoint_penalty(mixed_data2, r_data, y_mixed2, y_r)
                + self.endpoint_penalty(g_data, mixed_data2, y_g, y_mixed2)
                + self.endpoint_penalty(mixed_data1, mixed_data2, y_mixed1, y_mixed2))
    # loss, backprop, update
    loss = ep_penalty.mean() + y_r.mean() - y_g.mean()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self, cond):
    self.optim_g.zero_grad()
    g_data = self.eval_gen(cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.eval_disc(cond, g_data)
    loss = y_g.mean()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def endpoint_penalty(self, x1, x2, y1, y2):
    dist = torch.sqrt(((x1 - x2)**2).mean(1))
    # one-sided L1 penalty:
    penalty = F.relu(torch.abs(y1 - y2)/(dist*k_L + 1e-6) - 1.)
    # weight by square root of separation
    return torch.sqrt(dist)*penalty
  def eval_gen(self, cond):
    batch = cond.shape[0]
    ans = []
    for i in range(batch):
      noise = NOISE_IRREPS.randn(12, -1, device=self.config.device)
      tg_cond = torch_geometric.data.Data(x=noise, pos=cond[i], y=None)
      delta = self.gen(tg_cond)
      ans.append(cond[i] + delta)
    return torch.stack(ans, dim=0)
  def eval_disc(self, cond, data):
    batch = cond.shape[0]
    ans = []
    for i in range(batch):
      delta = data[i] - cond[i]
      tg_cond = torch_geometric.data.Data(x=delta, pos=cond[i], y=None)
      y = self.disc(tg_cond)
      ans.append(y)
    return torch.stack(ans, dim=0)
  def predict(self, cond):
    with torch.no_grad():
      ans = self.eval_gen(cond.reshape(-1, 12, 3))
      return ans.reshape(-1, 36)
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
modelclass   = GAN
trainerclass = GANTrainer



