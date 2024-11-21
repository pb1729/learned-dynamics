import torch
import torch.nn as nn
import torch.nn.functional as F

from polymer_util import poly_len, space_dim
from utils import must_be, prod
from layers_common import *
from attention_layers import *
from config import Config
from tensor_products import AVDFullLinearMix, AVDFullTensorProds, TensLinear, TensorRandGen, TensConv1d, TensGroupNorm, tens_sigmoid, TensTrace


class PosEmbed(nn.Module):
  """ Embeddings of node-wise relative positions. """
  def __init__(self, dim):
    super().__init__()
    self.lin = VecLinear(3, dim)
  def forward(self, pos_0, pos_1, pos_2):
    """ pos_0, pos_1, pos_2: (batch, nodes, 3) """
    pos_0, pos_1, pos_2 = pos_0[:, :, None], pos_1[:, :, None], pos_2[:, :, None]
    return self.lin(torch.cat([
      pos_1 - pos_0,
      pos_2 - pos_1,
      pos_0 - pos_2], dim=2))

class ArcEmbed(nn.Module):
  """ gets cosine embeddings for node position along the arc of the chain """
  def __init__(self, adim, no_const=True):
    super().__init__()
    self.adim = adim
    self.w_index_offset = float(no_const)
  def forward(self, batch, nodes, device):
    """ ans: (batch, nodes, adim) """
    s = torch.linspace(0., 1., nodes, device=device)[None, :, None].expand(batch, -1, -1)
    w = torch.pi*(self.w_index_offset + torch.arange(self.adim, device=device))[None, None, :]
    return torch.cos(s*w)

class Block(nn.Module):
  """ Generic block for generator and discriminator nets. """
  def __init__(self, z_scale:float, config:Config, randgen:TensorRandGen):
    super().__init__()
    self.z_scale = z_scale
    self.randgen = randgen
    dim_a, dim_v, dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    # submodules:
    self.arc_embed = ArcEmbed(dim_a)
    self.pos_embed = PosEmbed(dim_v)
    self.rand_lin_a = TensLinear(0, dim_a, dim_a)
    self.rand_lin_v = TensLinear(1, dim_v, dim_v)
    self.rand_lin_d = TensLinear(2, dim_d, dim_d)
    self.conv0_a = TensConv1d(0, dim_a, 7)
    self.conv0_v = TensConv1d(1, dim_v, 7)
    self.conv0_d = TensConv1d(2, dim_d, 7)
    self.conv1_a = TensConv1d(0, dim_a, 7)
    self.conv1_v = TensConv1d(1, dim_v, 7)
    self.conv1_d = TensConv1d(2, dim_d, 7)
    self.conv2_a = TensConv1d(0, dim_a, 7)
    self.conv2_v = TensConv1d(1, dim_v, 7)
    self.conv2_d = TensConv1d(2, dim_d, 7)
    self.linear_mix = AVDFullLinearMix(dim_a, dim_v, dim_d)
    self.tens_prods = AVDFullTensorProds(dim_a, dim_v, dim_d, config["rank"])
    self.lin_push_pos = TensLinear(1, dim_v, 1)
    self.probe_pts_q = ProbePoints3(len(config["r0_list"]), dim_v)
    self.probe_pts_k = ProbePoints3(len(config["r0_list"]), dim_v)
    self.prox_attn = ProximityFlashAttention(config["r0_list"], dim_a, dim_v, 16, vec_actv_class=VectorSigmoid)
    self.gn_a = TensGroupNorm(0, dim_a, config["groups_a"])
    self.gn_v = TensGroupNorm(1, dim_v, config["groups_v"])
    self.gn_d = TensGroupNorm(2, dim_d, config["groups_d"])
  def forward(self, tup):
    pos_0, pos_1, pos_2, x_a, x_v, x_d = tup
    # residual connections
    r_a, r_v, r_d = x_a, x_v, x_d
    # random noising of pos_2
    pos_2 = pos_2 + self.z_scale*self.randgen.randn(1, pos_0.shape[:-1])
    # embed positions (TODO in future archs: add ACE embedding too!)
    x_a = x_a + self.arc_embed(pos_0.shape[0], pos_0.shape[1], pos_0.device)
    x_v = x_v + self.pos_embed(pos_0, pos_1, pos_2)
    # add noise
    x_a = x_a + self.randgen.randn(0, x_a.shape)
    x_v = x_v + self.randgen.randn(1, x_v.shape[:-1])
    x_d = x_d + self.randgen.randn(2, x_d.shape[:-2])
    # convolutional MLP
    x_a = self.conv2_a(torch.relu(self.conv1_a(torch.relu(self.conv0_a(x_a))))).contiguous()
    x_v = self.conv2_v(tens_sigmoid(1, self.conv1_v(tens_sigmoid(1, self.conv0_v(x_v))))).contiguous()
    x_d = self.conv2_d(tens_sigmoid(2, self.conv1_d(tens_sigmoid(2, self.conv0_d(x_d))))).contiguous()
    # linear mix
    x_a, x_v, x_d = self.linear_mix(x_a, x_v, x_d)
    # tensor products
    Δx_a, Δx_v, Δx_d = self.tens_prods(x_a, x_v, x_d)
    x_a, x_v, x_d = x_a + Δx_a, x_v + Δx_v, x_d + Δx_d
    # push pos_2 by some vectors
    pos_2 = pos_2 + self.lin_push_pos(x_v)[:, :, 0]
    # flash attention
    probes_q = self.probe_pts_q(x_v, pos_2)
    probes_k = self.probe_pts_k(x_v, pos_2)
    Δx_a, Δx_v = self.prox_attn(x_a, x_v, probes_k, probes_q)
    x_a, x_v = x_a + Δx_a, x_v + Δx_v
    # group norm
    x_a, x_v, x_d = self.gn_a(x_a), self.gn_v(x_v), self.gn_d(x_d)
    return pos_0, pos_1, pos_2, r_a + x_a, r_v + x_v, r_d + x_d


class Discriminator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.blocks = nn.Sequential(*[
      Block(z_scale, config, randgen)
      for z_scale in config["z_scale"]
    ])
    self.lin_a = nn.Linear(self.dim_a, config["heads"], bias=False)
    self.lin_v = VecLinear(self.dim_v, config["heads"])
    self.lin_d = TensTrace(2, self.dim_d, config["heads"])
  def forward(self, pos_0, pos_1):
    device = pos_0.device
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = torch.zeros(batch, nodes, self.dim_v, 3, device=device)
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    pos_0, pos_1, pos_2, x_a, x_v, x_d = self.blocks((pos_0, pos_1, pos_0, x_a, x_v, x_d))
    # use taxicab distance of pos_1 and pos_2 as part of discriminator output
    y = taxicab(pos_0, pos_2)[:, None] # (batch, 1)
    # activation-based output, with distinct heads
    y = y + self.lin_a(x_a.mean(1))                           # (batch, heads)
    y = y + torch.sqrt((self.lin_v(x_v.mean(1))**2).sum(-1))  # (batch, heads)
    y = y + self.lin_d(x_d.mean(1))                           # (batch, heads)
    return y


class Generator(nn.Module):
  def __init__(self, config, randgen):
    super().__init__()
    self.dim_a, self.dim_v, self.dim_d = config["dim_a"], config["dim_v"], config["dim_d"]
    self.blocks = nn.Sequential(*[
      Block(z_scale, config, randgen)
      for z_scale in config["z_scale"]
    ])
  def forward(self, pos_0):
    device = pos_0.device
    batch,          nodes,          must_be[3] = pos_0.shape
    x_a = torch.zeros(batch, nodes, self.dim_a, device=device)
    x_v = torch.zeros(batch, nodes, self.dim_v, 3, device=device)
    x_d = torch.zeros(batch, nodes, self.dim_d, 3, 3, device=device)
    # run the main network
    pos_0, pos_1, pos_2, x_a, x_v, x_d = self.blocks((pos_0, pos_0, pos_0, x_a, x_v, x_d))
    return pos_2


def taxicab(x1, x2, epsilon = 0.01):
  """ x1, x2: (batch, poly_len, 3)
      ans: (batch) """
  return torch.sqrt(((x1 - x2)**2).sum(-1) + epsilon).mean(-1)
def endpoint_penalty(x1, x2, y1, y2):
  """ functional form of endpoint penalty
      x1, x2: (batch, poly_len, 3)
      y1, y2: (batch, heads)
      ans: () """
  # use the taxicab metric (take average distance that nodes moved rather than RMS distance)
  dist = taxicab(x1, x2)[:, None] # (batch, 1)
  # one-sided L1 penalty:
  penalty_l1 = F.relu(torch.abs(y1 - y2)/dist - 1.) # (batch, heads)
  # zero-centered L2 penalty:
  penalty_l2 = 0.2*((y1 - y2)/dist)**2 # (batch, heads)
  return penalty_l1.mean() + penalty_l2.mean()
def get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g):
  """ full computation of endpoint penalty on interpolated data
      x_0: (batch, poly_len, 3)
      x_r, x_g: (batch, poly_len, 3)
      y_r, y_g: (batch, heads)
      ans: () """
  batch, nodes, must_be[3] = x_0.shape
  assert x_r.shape == x_0.shape == x_g.shape
  mix_factors_1 = torch.rand(batch, 1, 1, device=x_0.device)
  mix_factors_2 = torch.rand(batch, 1, 1, device=x_0.device)
  x_1 = mix_factors_1*x_g + (1 - mix_factors_1)*x_r
  x_2 = mix_factors_2*x_g + (1 - mix_factors_2)*x_r
  y_1 = disc(x_0, x_1)
  y_2 = disc(x_0, x_2)
  return (endpoint_penalty(x_r, x_g, y_r, y_g)
        + endpoint_penalty(x_r, x_1, y_r, y_1)
        + endpoint_penalty(x_r, x_2, y_r, y_2)
        + endpoint_penalty(x_g, x_1, y_g, y_1)
        + endpoint_penalty(x_g, x_2, y_g, y_2)
        + endpoint_penalty(x_1, x_2, y_1, y_2))


class WGAN3D:
  is_gan = True
  def __init__(self, config):
    self.randgen = TensorRandGen()
    self.discs = [Discriminator(config, self.randgen).to(config.device) for _ in range(config["ndiscs"])]
    self.gen  = Generator(config, self.randgen).to(config.device)
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
    ans = WGAN3D(config)
    for disc, state_dict in zip(ans.discs, states["discs"]):
      disc.load_state_dict(state_dict)
    ans.gen.load_state_dict(states["gen"])
    return ans
  @staticmethod
  def makenew(config):
    ans = WGAN3D(config)
    for disc in ans.discs:
      disc.apply(weights_init)
    ans.gen.apply(weights_init)
    return ans
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
    endpt_pen = get_endpt_pen(disc, x_0, x_r, x_g, y_r, y_g)
    # overall loss
    if self.config["hinge"]:
      loss = torch.relu(1. + y_r).mean() + torch.relu(1. - y_g).mean() + self.config["hinge_leak"]*(y_r.mean() - y_g.mean())
    else:
      loss = y_r.mean() - y_g.mean()
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
  def generate(self, x_0):
    *leading_dims, must_be[self.n_nodes], must_be[3] = x_0.shape
    batch = prod(leading_dims)
    x_0 = x_0.reshape(batch, self.n_nodes, 3)
    ans = self.gen(x_0)
    return ans.reshape(*leading_dims, self.n_nodes, 3)
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
