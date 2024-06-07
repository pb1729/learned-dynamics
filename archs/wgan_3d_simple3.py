import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn

from config import Condition
from utils import must_be
from gan_common import GANTrainer
from layers_common import *


# constants:
k_L = 1.       # Lipschitz constant


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
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
    self.bn_a_nodes = e3nn.nn.BatchNorm(f"{adim}x0e")
    self.bn_v_nodes = e3nn.nn.BatchNorm(f"{vdim}x1o")
    self.bn_a_edges = e3nn.nn.BatchNorm(f"{adim}x0e")
    self.bn_v_edges = e3nn.nn.BatchNorm(f"{vdim}x1o")
  def forward(self, tup):
    a_node, v_node, a_edge, v_edge, graph = tup
    # pre conv:
    x_a_node = self.a_nodes_conv_0(a_node, graph)
    x_v_node = self.v_nodes_conv_0(v_node, graph)
    x_a_edge = a_edge + self.a_edges_read(x_a_node, graph)
    x_v_edge = v_edge + self.v_edges_read(x_v_node, graph)
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
    # batchnorms:
    batch,          nodes,          vdim, must_be[3] = z_v_node.shape
    must_be[batch], edges, must_be[vdim], must_be[3] = z_v_edge.shape
    z_a_node = self.bn_a_nodes(z_a_node)
    z_v_node = self.bn_v_nodes(z_v_node.reshape(batch, nodes, 3*vdim)).reshape(batch, nodes, vdim, 3)
    z_a_edge = self.bn_a_edges(z_a_edge)
    z_v_edge = self.bn_v_edges(z_v_edge.reshape(batch, edges, 3*vdim)).reshape(batch, edges, vdim, 3)
    # residual-style result:
    out_a_node = a_node + z_a_node
    out_v_node = v_node + z_v_node
    out_a_edge = a_edge + z_a_edge
    out_v_edge = v_edge + z_v_edge
    return (out_a_node, out_v_node, out_a_edge, out_v_edge, graph)




class Discriminator(nn.Module):
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
    self.a_edges_write = ScalEdgesWrite(adim, adim)
    self.v_edges_write = VecEdgesWrite(vdim, vdim)
    self.lin_a_node = nn.Linear(adim, 1)
    self.lin_v_node = nn.Linear(vdim, 1)
  def forward(self, pos0, pos1, graph):
    a_node, v_node = self.node_enc(pos0, pos1, graph)
    a_edge, v_edge = self.edge_enc(pos0, pos1, graph)
    a_node, v_node, a_edge, v_edge, _ = self.blocks((a_node, v_node, a_edge, v_edge, graph))
    a_node = (a_node + self.a_edges_write(a_edge, graph))*INV_SQRT_2
    v_node = (v_node + self.v_edges_write(v_edge, graph))*INV_SQRT_2
    v_norms = torch.linalg.vector_norm(v_node, dim=-1)
    return (self.lin_a_node(a_node) + self.lin_v_node(v_norms)).sum(2).mean(1)


class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.edge_enc = EdgeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.v_edges_write = VecEdgesWrite(vdim, vdim)
    self.lin_v_node = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def _predict(self, pos0, noised, z_a, z_v, graph):
    a_node, v_node = self.node_enc(pos0, noised, graph)
    a_edge, v_edge = self.edge_enc(pos0, noised, graph)
    a_node = a_node + z_a
    v_node = v_node + z_v
    a_node, v_node, a_edge, v_edge, _ = self.blocks((a_node, v_node, a_edge, v_edge, graph))
    v_node = (v_node + self.v_edges_write(v_edge, graph))*INV_SQRT_2
    return self.lin_v_node(v_node)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, noise, z_a, z_v, graph):
    noised = pos0 + noise
    pred_noise = self._predict(pos0, noised, z_a, z_v, graph)
    return noised - pred_noise



class WGAN3D:
  is_gan = True
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    assert config.sim.space_dim == 3
    assert config.cond_type == Condition.COORDS
    assert config.subtract_mean == 0
    self.init_optim()
    self.init_graph()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.Adam(self.disc.parameters(), self.config["lr_d"], betas)
    self.optim_g = torch.optim.Adam(self.gen.parameters(),  self.config["lr_g"], betas)
  def init_graph(self):
    n_nodes = self.config.sim.poly_len
    r = torch.arange(n_nodes - 1, device=self.config.device)
    src = torch.cat([r, r + 1])
    dst = torch.cat([r + 1, r])
    self.graph = Graph(src, dst, n_nodes)
  @staticmethod
  def load_from_dict(states, config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return WGAN3D(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return WGAN3D(disc, gen, config)
  def save_to_dict(self):
    return {
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    data, cond = data.reshape(-1, self.graph.n_nodes, 3), cond.reshape(-1, self.graph.n_nodes, 3)
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, data, cond):
    self.optim_d.zero_grad()
    # train on real data (with instance noise)
    instance_noise_r = self.config["inst_noise_str_r"]*torch.randn_like(data)
    r_data = data + instance_noise_r # instance noise
    y_r = self.disc(cond, r_data, self.graph)
    # train on generated data (with instance noise)
    g_data = self.generate(cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.disc(cond, g_data, self.graph)
    # endpoint penalty on interpolated data
    mix_factors1 = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    mixed_data1 = mix_factors1*g_data + (1 - mix_factors1)*r_data
    y_mixed1 = self.disc(cond, mixed_data1, self.graph)
    mix_factors2 = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    mixed_data2 = mix_factors2*g_data + (1 - mix_factors2)*r_data
    y_mixed2 = self.disc(cond, mixed_data2, self.graph)
    ep_penalty = (self.endpoint_penalty(r_data, g_data, y_r, y_g)
                + self.endpoint_penalty(mixed_data1, r_data, y_mixed1, y_r)
                + self.endpoint_penalty(g_data, mixed_data1, y_g, y_mixed1)
                + self.endpoint_penalty(mixed_data2, r_data, y_mixed2, y_r)
                + self.endpoint_penalty(g_data, mixed_data2, y_g, y_mixed2)
                + self.endpoint_penalty(mixed_data1, mixed_data2, y_mixed1, y_mixed2))
    # loss, backprop, update
    loss = self.config["lpen_wt"]*ep_penalty.mean() + y_r.mean() - y_g.mean()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self, cond):
    self.optim_g.zero_grad()
    g_data = self.generate(cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.disc(cond, g_data, self.graph)
    loss = y_g.mean()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def endpoint_penalty(self, x1, x2, y1, y2):
    dist = torch.sqrt(((x1 - x2)**2).sum(2).mean(1)) # average across nodes to make scaling easier
    # one-sided L1 penalty:
    penalty = F.relu(torch.abs(y1 - y2)/(dist*k_L + 1e-6) - 1.)
    # weight by square root of separation
    return torch.sqrt(dist)*penalty
  def generate(self, cond):
    batch, must_be[self.graph.n_nodes], must_be[3] = cond.shape
    pos_noise, z_a, z_v = self.get_latents(batch)
    return self.gen(cond, pos_noise, z_a, z_v, self.graph)
  def get_latents(self, batchsz):
    """ sample latent space for generator """
    pos_noise = self.config["z_scale"]*torch.randn(batchsz, self.graph.n_nodes, 3, device=self.config.device)
    z_a = torch.randn(batchsz, self.graph.n_nodes, self.config["adim"], device=self.config.device)
    z_v = torch.randn(batchsz, self.graph.n_nodes, self.config["vdim"], 3, device=self.config.device)
    return pos_noise, z_a, z_v
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()
  def predict(self, cond):
    batch, must_be[3*self.graph.n_nodes] = cond.shape
    with torch.no_grad():
      return self.generate(cond.reshape(batch, self.graph.n_nodes, 3)).reshape(batch, 3*self.graph.n_nodes)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer



