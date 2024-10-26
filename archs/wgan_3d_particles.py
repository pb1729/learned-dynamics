import torch
import torch.nn as nn
import torch.nn.functional as F

from polymer_util import poly_len, space_dim
from utils import must_be, prod
from layers_common import *
from attention_layers import *
from config import Config
from graph_layers import Graph, edges_read, edges_reduce


class GraphEmbedLayer(nn.Module):
  def __init__(self, r_cut):
    super().__init__()
    self.r_cut = r_cut
  def forward(self, graph:Graph, pos):
    """ pos: (batch, nodes, 3) """
    batch, nodes, must_be[3] = pos.shape
    ang_box = torch.tensor(graph.box, device=pos.device)/(2*torch.pi) # box, angular frequency version
    edges = edges_read(graph, pos) - pos[:, :, None] # (batch, nodes, neighbours, 3)
    edges = ang_box*torch.sin(edges/ang_box)
    r = (edges**2).sum(-1) # (batch, nodes, neighbours)
    x = radial_encode(r, 8, self.r_cut) # (batch, nodes, neighbours, 8)
    xv = x[..., None]*edges[..., None, :] # (batch, nodes, neighbours, 8, 3)
    #xvv = x[..., None, None]*edges[..., None, :, None]*edges[..., None, None, :] # (batch, nodes, neighbours, 8, 3, 3)
    y = edges_reduce(graph, x)
    yv = edges_reduce(graph, xv.reshape(batch, nodes, -1, 8*3)).reshape(batch, nodes, 8, 3)
    #yvv = edges_reduce(graph, xvv.reshape(batch, nodes, -1, 8*3*3)).reshape(batch, nodes, 8, 3)
    return y, yv


class LocalResidual(nn.Module):
  """ Residual layer that just does local processing on individual nodes. """
  def __init__(self, config):
    super().__init__()
    adim, vdim, rank = config["adim"], config["vdim"], config["rank"]
    agroups, vgroups = config["agroups"], config["vgroups"]
    self.layers_a = nn.Sequential(
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.lin_v = VecLinear(vdim, vdim)
    self.av_prod = ScalVecProducts(adim, vdim, rank)
    self.gnorm_a = ScalGroupNorm(adim, agroups)
    self.gnorm_v = VecGroupNorm(vdim, vgroups)
  def forward(self, x_a, x_v):
    y_a = self.layers_a(x_a)
    y_v = self.lin_v(x_v)
    p_a, p_v = self.av_prod(x_a, x_v)
    z_a, z_v = self.gnorm_a(y_a + p_a), self.gnorm_v(y_v + p_v)
    return (x_a + z_a), (x_v + z_v)


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    agroups, vgroups = config["agroups"], config["vgroups"]
    self.node_embed = GraphEmbedLayer(config["r_cut"])
    self.emb_a = nn.Linear(8, adim)
    self.emb_v = VecLinear(8, vdim)
    self.local_res = LocalResidual(config)
    self.probe_pts_q = ProbePoints3(len(config["r0_list"]), vdim)
    self.probe_pts_k = ProbePoints3(len(config["r0_list"]), vdim)
    self.prox_attn = ProximityFlashAttentionPeriodic(config["r0_list"], adim, vdim, 16)
    self.gnorm_a = ScalGroupNorm(adim, agroups)
    self.gnorm_v = VecGroupNorm(vdim, vgroups)
  def forward(self, tup):
    box, pos_0, pos_1, graph, x_a, x_v = tup
    emb_a, emb_v = self.node_embed(graph, pos_0)
    y_a = x_a + self.emb_a(emb_a)
    y_v = x_v + self.emb_v(emb_v)
    z_a, z_v = self.local_res(y_a, y_v)
    probes_q = self.probe_pts_q(z_v, pos_0)
    probes_k = self.probe_pts_k(z_v, pos_0)
    dz_a, dz_v = self.prox_attn(z_a, z_v, probes_k, probes_q, box)
    z_a, z_v = z_a + dz_a, z_v + dz_v
    return box, pos_0, pos_1, graph, (x_a + y_a), (x_v + y_v)


class Discriminator(nn.Module):
  def __init__(self, config:Config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.r_cut = config["r_cut"]
    box = config.predictor.get_box()
    self.box = (box[0].item(), box[1].item(), box[2].item())
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    self.lin_out = nn.Linear(adim, config["heads"])
  def forward(self, pos_0, pos_1):
    graph = Graph.radius_graph(self.r_cut, self.box, pos_0, neighbours_max=64)
    x_a, x_v = self.node_enc(pos_0, pos_1)
    *_, y_a, y_v = self.blocks((self.box, pos_0, pos_1, graph, x_a, x_v))
    return self.lin_out(y_a.mean(1))


class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.r_cut = config["r_cut"]
    box = config.predictor.get_box()
    self.box = (box[0].item(), box[1].item(), box[2].item())
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.lin_v_node = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-1)
  def _predict(self, pos0, noised):
    graph = Graph.radius_graph(self.r_cut, self.box, pos0, neighbours_max=64)
    x_a, x_v = self.node_enc(pos0, noised)
    *_, y_v = self.blocks((self.box, pos0, noised, graph, x_a, x_v))
    return self.lin_v_node(y_v)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, noise, ε_a, ε_v):
    noised = pos0 + noise
    pred_noise = self._predict(pos0, noised)
    return noised - pred_noise


class WGAN3D:
  is_gan = True
  def __init__(self, discs, gen, config):
    self.discs = discs
    self.gen  = gen
    self.config = config
    assert space_dim(config) == 3
    self.n_nodes = poly_len(config)
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.AdamW((param for disc in self.discs for param in disc.parameters()),
      self.config["lr_d"], betas, weight_decay=self.config["weight_decay"])
    self.optim_g = torch.optim.AdamW(self.gen.parameters(),
      self.config["lr_g"], betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
  @staticmethod
  def load_from_dict(states, config):
    discs = [Discriminator(config).to(config.device) for _ in range(config["ndiscs"])]
    gen = Generator(config).to(config.device)
    for disc, state_dict in zip(discs, states["discs"]):
      disc.load_state_dict(state_dict)
    gen.load_state_dict(states["gen"])
    return WGAN3D(discs, gen, config)
  @staticmethod
  def makenew(config):
    discs = [Discriminator(config).to(config.device) for _ in range(config["ndiscs"])]
    gen = Generator(config).to(config.device)
    for disc in discs:
      disc.apply(weights_init)
    gen.apply(weights_init)
    return WGAN3D(discs, gen, config)
  def save_to_dict(self):
    return {
        "discs": [disc.state_dict() for disc in self.discs],
        "gen": self.gen.state_dict(),
      }
  def train_step(self, x):
    """ x: (batch, L, poly_len, 3) """
    batch, L, must_be[self.n_nodes], must_be[3] = x.shape
    loss_d = self.discs_step(x)
    loss_g = self.gen_step(x)
    self.step_count += 1
    if self.step_count % 1024 == 0:
      self.lr_schedule_update()
    return loss_d, loss_g
  def lr_schedule_update(self):
    try:
      lr_d_fac = self.config["lr_d_fac"]
      lr_g_fac = self.config["lr_g_fac"]
    except IndexError:
      lr_d_fac = 0.99
      lr_g_fac = 0.95
    for group in self.optim_g.param_groups: # learning rate schedule
      group["lr"] *= lr_g_fac
    for group in self.optim_d.param_groups: # learning rate schedule
      group["lr"] *= lr_d_fac
  def discs_step(self, x):
    """ x: (batch, L, poly_len, 3) """
    batch, L, must_be[self.n_nodes], must_be[3] = x.shape
    x_g = x
    loss = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:, :-1])
      x_r = x[:, nsteps:]
      x_0 = x[:, :-nsteps]
      loss = loss + self.disc_loss(disc,
        x_0.reshape(batch*(L - nsteps), self.n_nodes, 3),
        x_r.reshape(batch*(L - nsteps), self.n_nodes, 3),
        x_g.reshape(batch*(L - nsteps), self.n_nodes, 3))
    # backprop, update
    self.optim_d.zero_grad()
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def disc_loss(self, disc, x_0, x_r, x_g):
    batch,          must_be[self.n_nodes], must_be[3] = x_0.shape
    must_be[batch], must_be[self.n_nodes], must_be[3] = x_r.shape
    must_be[batch], must_be[self.n_nodes], must_be[3] = x_g.shape
    # train on real data
    y_r = disc(x_0, x_r) # (batch, heads)
    # train on generated data
    y_g = disc(x_0, x_g) # (batch, heads)
    # endpoint penalty on interpolated data
    endpt_pen = self.get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g)
    # overall loss
    if self.config["hinge"]:
      loss = torch.relu(1. + y_r).mean() + torch.relu(1. - y_g).mean() + self.config["hinge_leak"]*(y_r.mean() - y_g.mean())
    else:
      loss = y_r.mean() - y_g.mean()
    # covariance loss
    if self.config["covar_pen"]:
      C = torch.cov((y_r - y_g).T) # (heads, heads)
      loss_cov = (C**2 - torch.diag(torch.diag(C)**2)).mean()
      loss = loss + loss_cov
    return loss + self.config["lpen_wt"]*endpt_pen
  def gen_step(self, x):
    x_g = x
    loss = 0.
    for nsteps, disc in enumerate(self.discs, start=1):
      x_g = self.generate(x_g[:, :-1])
      x_0 = x[:, :-nsteps]
      y_g = disc(x_0.reshape(-1, self.n_nodes, 3), x_g.reshape(-1, self.n_nodes, 3))
      loss = loss + y_g.mean()
    # backprop, update
    self.optim_g.zero_grad()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def get_endpt_pen(self, disc, x_0, x_r, x_g, y_r, y_g):
    """ full computation of endpoint penalty on interpolated data
    x_0: (batch, poly_len, 3)
    x_r, x_g: (batch, poly_len, 3)
    y_r, y_g: (batch, 1)
    ans: () """
    batch, must_be[self.n_nodes], must_be[3] = x_0.shape
    assert x_r.shape == x_0.shape == x_g.shape
    mix_factors_1 = torch.rand(batch, 1, 1, device=self.config.device)
    mix_factors_2 = torch.rand(batch, 1, 1, device=self.config.device)
    x_1 = mix_factors_1*x_g + (1 - mix_factors_1)*x_r
    x_2 = mix_factors_2*x_g + (1 - mix_factors_2)*x_r
    y_1 = disc(x_0, x_1)
    y_2 = disc(x_0, x_2)
    return (self.endpoint_penalty(x_r, x_g, y_r, y_g)
          + self.endpoint_penalty(x_r, x_1, y_r, y_1)
          + self.endpoint_penalty(x_r, x_2, y_r, y_2)
          + self.endpoint_penalty(x_g, x_1, y_g, y_1)
          + self.endpoint_penalty(x_g, x_2, y_g, y_2)
          + self.endpoint_penalty(x_1, x_2, y_1, y_2))
  def endpoint_penalty(self, x1, x2, y1, y2):
    """ functional form of endpoint penalty
    x1, x2: (batch, poly_len, 3)
    y1, y2: (batch, 1)
    ans: () """
    epsilon = 0.01 # this will be put into distance, since we'll be dividing by it
    # use the taxicab metric (take average distance that nodes moved rather than RMS distance)
    dist = torch.sqrt(((x1 - x2)**2).sum(-1) + epsilon).mean(-1)[:, None] # (batch, heads)
    # one-sided L1 penalty:
    penalty_l1 = F.relu(torch.abs(y1 - y2)/dist - 1.) # (batch, heads)
    # zero-centered L2 penalty:
    penalty_l2 = 0.2*((y1 - y2)/dist)**2 # (batch, heads)
    return penalty_l1.mean() + penalty_l2.mean()
  def generate(self, x_0):
    *leading_dims, must_be[self.n_nodes], must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, self.n_nodes, 3)
    pos_noise, z_a, z_v = self.get_latents(batch)
    ans = self.gen(x_0, pos_noise, z_a, z_v)
    return ans.reshape(*leading_dims, self.n_nodes, 3)
  def get_latents(self, batchsz):
    """ sample latent space for generator """
    pos_noise = self.config["z_scale"]*torch.randn(batchsz, self.n_nodes, 3, device=self.config.device)
    z_a = torch.randn(batchsz, self.n_nodes, self.config["adim"], device=self.config.device)
    z_v = torch.randn(batchsz, self.n_nodes, self.config["vdim"], 3, device=self.config.device)
    return pos_noise, z_a, z_v
  def set_eval(self, bool_eval):
    if bool_eval:
      for disc in self.discs:
        disc.eval()
      self.gen.eval()
    else:
      for disc in self.discs:
        disc.train()
      self.gen.train()
  def predict(self, cond):
    with torch.no_grad():
      return self.generate(cond)


class GANTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    config = self.model.config
    loss_d, loss_g = self.model.train_step(trajs)
    print(f"{i}\t ℒᴰ = {loss_d:05.6f}   \t ℒᴳ = {loss_g:05.6f}")
    self.board.scalar("loss_d", i, loss_d)
    self.board.scalar("loss_g", i, loss_g)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer
