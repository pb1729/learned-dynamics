import torch
import torch.nn as nn

from config import Condition
from utils import must_be
from layers_common import *


def g_cos(t, device):
  """ this function implicitly defines the noise schedule
      this particular function gives the cosine noise schedule
      read more here: https://arxiv.org/pdf/2102.09672
      t is a number in the range [0, 1], indicating where in the diffusion process we are
      returns g(t) """
  if isinstance(t, float): t = torch.tensor(t, device=device)
  s = 0.008
  t_adj = (t + s)/(1 + s) # adjusted time, see explanation after eq. 17 on pg. 4 of the paper
  return torch.cos(0.5*torch.pi*t_adj)**2


class TimeEmbedding(nn.Module):
  def __init__(self, hdim, outdim):
    super(TimeEmbedding, self).__init__()
    self.hdim = hdim
    self.lin1 = nn.Linear(2*self.hdim, outdim)
  def raw_t_embed(self, t):
    """ t: (batch)
        return: (batch, 1, hdim) """
    ang_freqs = torch.exp(-torch.arange(self.hdim, device=t.device)/(self.hdim - 1))
    phases = t[:, None] * ang_freqs[None, :]
    return torch.cat([
      torch.sin(phases),
      torch.cos(phases),
    ], dim=1)[:, None]
  def forward(self, t):
    """ t: (batch)
        return: (batch, 1, outdim) """
    return self.lin1(self.raw_t_embed(t))


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    agroups, vgroups = config["agroups"], config["vgroups"]
    self.t_embed = TimeEmbedding(config["time_hdim"], adim)
    self.a_nodes_conv_0 = ScalNodesConv(adim, adim)
    self.v_nodes_conv_0 = VecNodesConv(vdim, vdim)
    self.a_edges_read = ScalEdgesRead(adim, adim)
    self.v_edges_read = VecEdgesRead(vdim, vdim)
    self.a_node_layers = nn.Sequential(
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.a_edge_layers = nn.Sequential(
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.v_node_layers = nn.Sequential(
      VecLinear(vdim, vdim),
      VecRootS(),
      VecLinear(vdim, vdim))
    self.v_edge_layers = nn.Sequential(
      VecLinear(vdim, vdim),
      VecRootS(),
      VecLinear(vdim, vdim))
    self.a_res = Residual(adim)
    self.v_res = VecResidual(vdim)
    self.av_node_prod = ScalVecProducts(adim, vdim, config["rank"])
    self.av_edge_prod = ScalVecProducts(adim, vdim, config["rank"])
    self.a_edges_write = ScalEdgesWrite(adim, adim)
    self.v_edges_write = VecEdgesWrite(vdim, vdim)
    self.a_nodes_conv_1 = ScalNodesConv(adim, adim)
    self.v_nodes_conv_1 = VecNodesConv(vdim, vdim)
    self.gn_a_nodes = ScalGroupNorm(adim, agroups)
    self.gn_v_nodes = VecGroupNorm(vdim, vgroups)
    self.gn_a_edges = ScalGroupNorm(adim, agroups)
    self.gn_v_edges = VecGroupNorm(vdim, vgroups)
  def forward(self, tup):
    a_node, v_node, a_edge, v_edge, graph, t = tup
    # pre conv:
    x_a_node = self.a_nodes_conv_0(a_node, graph)
    x_v_node = self.v_nodes_conv_0(v_node, graph)
    x_a_edge = a_edge + self.a_edges_read(x_a_node, graph)
    x_v_edge = v_edge + self.v_edges_read(x_v_node, graph)
    # mix in the time embedding
    x_a_node = x_a_node + self.t_embed(t)
    # non-linearities:
    y_a_node = self.a_node_layers(x_a_node)
    y_a_edge = self.a_edge_layers(x_a_edge)
    y_v_node = self.v_node_layers(x_v_node)
    y_v_edge = self.v_edge_layers(x_v_edge)
    prod_a_node, prod_v_node = self.av_node_prod(x_a_node, x_v_node)
    prod_a_edge, prod_v_edge = self.av_node_prod(x_a_edge, x_v_edge)
    z_a_node, z_v_node = y_a_node + prod_a_node, y_v_node + prod_v_node
    z_a_edge, z_v_edge = y_a_edge + prod_a_edge, y_v_edge + prod_v_edge
    # post conv:
    z_a_node = z_a_node + self.a_edges_write(z_a_edge, graph)
    z_v_node = z_v_node + self.v_edges_write(z_v_edge, graph)
    z_a_node = self.a_nodes_conv_1(z_a_node, graph)
    z_v_node = self.v_nodes_conv_1(z_v_node, graph)
    # groupnorms:
    z_a_node = self.gn_a_nodes(z_a_node)
    z_v_node = self.gn_v_nodes(z_v_node)
    z_a_edge = self.gn_a_edges(z_a_edge)
    z_v_edge = self.gn_v_edges(z_v_edge)
    # residual-style result:
    out_a_node = a_node + z_a_node
    out_v_node = v_node + z_v_node
    out_a_edge = a_edge + z_a_edge
    out_v_edge = v_edge + z_v_edge
    return (out_a_node, out_v_node, out_a_edge, out_v_edge, graph, t)


class Model(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.edge_enc = EdgeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.v_edges_write = VecEdgesWrite(vdim, vdim)
    self.lin_v_node = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def forward(self, pos0, noised, t, graph):
    """ pos0: (batch, nodes, 3)
        noised: (batch, nodes, 3)
        t: (batch) """
    a_node, v_node = self.node_enc(pos0, noised, graph)
    a_edge, v_edge = self.edge_enc(pos0, noised, graph)
    a_node, v_node, a_edge, v_edge, *_ = self.blocks((a_node, v_node, a_edge, v_edge, graph, t))
    v_node = (v_node + self.v_edges_write(v_edge, graph))*INV_SQRT_2
    return self.lin_v_node(v_node)[:, :, 0]*self.out_norm_coeff


class Diffuser3D:
  is_gan = True # not, really, but it tries to generate samples from a distribution, which is what we care about for plotting
  def __init__(self, model, config):
    self.model = model
    self.config = config
    assert config.sim.space_dim == 3
    assert config.cond_type == Condition.COORDS
    assert config.subtract_mean == 0
    self.init_optim()
    self.init_graph()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim = torch.optim.Adam(self.model.parameters(), self.config["lr"], betas)
  def init_graph(self):
    n_nodes = self.config.sim.poly_len
    r = torch.arange(n_nodes - 1, device=self.config.device)
    src = torch.cat([r, r + 1])
    dst = torch.cat([r + 1, r])
    self.graph = Graph(src, dst, n_nodes)
  @staticmethod
  def load_from_dict(states, config):
    model = Model(config).to(config.device)
    model.load_state_dict(states["model"])
    return Diffuser3D(model, config)
  @staticmethod
  def makenew(config):
    model = Model(config).to(config.device)
    model.apply(weights_init)
    return Diffuser3D(model, config)
  def save_to_dict(self):
    return {
        "model": self.model.state_dict(),
      }
  def train_step(self, data, cond):
    nodes = self.graph.n_nodes
    batch,          must_be[nodes*3] = data.shape
    must_be[batch], must_be[nodes*3] = cond.shape
    return self._train_step(cond.reshape(-1, nodes, 3), data.reshape(-1, nodes, 3))
  def _train_step(self, pos0, pos1):
    batch, nodes, must_be[3] = pos0.shape
    self.model.zero_grad()
    t = torch.rand(batch, device=self.config.device)
    sigma = self.config["z_scale"]*torch.sqrt(self.get_var(0.0, t))[:, None, None]
    noise = torch.randn_like(pos0)
    mu_coeff = self.get_mu_coeff(0.0, t)[:, None, None]
    noised = pos0 + sigma*noise + mu_coeff*(pos1 - pos0)
    pred_noise = self.model(pos0, noised, t, self.graph)
    loss = ((noise - pred_noise)**2).sum((1, 2)).mean()
    loss.backward()
    self.optim.step()
    return loss.item()
  def predict(self, cond, gen_steps=80):
    batch, must_be[3*self.graph.n_nodes] = cond.shape
    return self._predict(cond.reshape(batch, self.graph.n_nodes, 3), gen_steps).reshape(batch, 3*self.graph.n_nodes)
  def _predict(self, pos0, gen_steps):
    batch, nodes, must_be[3] = pos0.shape
    device = self.config.device
    with torch.no_grad():
      x = pos0.clone()
      for i in range(gen_steps - 1, 0, -1):
        s = torch.ones(batch, device=device)*(i-1)/gen_steps
        t = torch.ones(batch, device=device)*(i)/gen_steps
        sigma_fwd = self.config["z_scale"]*torch.sqrt(self.get_var(0.0, t))[:, None, None]
        x += sigma_fwd*torch.randn(batch, nodes, 3, device=device)
        pred_noise = self.model(pos0, x, t, self.graph)
        x -= sigma_fwd*pred_noise
        mu_coeff = self.get_mu_coeff(s, t)[:, None, None]
        x = pos0 + (x - pos0)/mu_coeff
      return x
  def set_eval(self, bool_eval):
    if bool_eval:
      self.model.eval()
    else:
      self.model.train()
  def get_var(self, s, t):
    """ Noise scheduling function: get the variance of p(x_t|x_s), the full matrix is ans*I
        g: float -> float : this function defines the noise schedule. it's the cumulative alpha product
        s: float          : start time
        t: float          : end time, larger than s """
    return 1. - self.G(t)/self.G(s)
  def get_mu_coeff(self, s, t):
    """ Noise scheduling function: get the decay in mean between times t and s.
        g: float -> float : this function defines the noise schedule. it's the cumulative alpha product
        s: float          : start time
        t: float          : end time, larger than s """
    return torch.sqrt(self.G(t)/self.G(s))
  def G(self, t): # define the noise schedule!
    return g_cos(t, self.config.device)


class DiffuserTrainer:
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
modelclass   = Diffuser3D
trainerclass = DiffuserTrainer



