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


# Main change in this file relative to wgan_3d_simple4 is that the graph structure is implicit in just
# a bunch of pure convolutions. This constrains the model to being used on a path graph, which is fine
# for now. This has the advantage that we can more easily do longer range convolutions, eg. with a kernel
# width of 5, or even 7. In the future, we might have a system consisting of many residues. At that point,
# we can have a two-level structure to the network, and the linear chain of residues can still have long
# range Conv1d's.


class ScalConv1d(nn.Module):
  def __init__(self, chan, kernsz, edges_to_nodes=False):
    super().__init__()
    if edges_to_nodes:
      assert kernsz % 2 == 0
      self.conv = nn.Conv1d(chan, chan, kernsz, padding=(kernsz//2))
    else:
      assert kernsz % 2 == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, padding="same")
  def forward(self, x):
    """ x: (batch, length, chan) """
    x = x.permute(0, 2, 1) # (batch, chan, length)
    y = self.conv(x)
    y = y.permute(0, 2, 1) # (batch, newlength, chan)
    return y

class VecConv1d(nn.Module):
  def __init__(self, chan, kernsz, edges_to_nodes=False):
    super().__init__()
    if edges_to_nodes:
      assert kernsz % 2 == 0
      self.conv = nn.Conv1d(chan, chan, kernsz, padding=(kernsz//2), bias=False)
    else:
      assert kernsz % 2 == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, padding="same", bias=False)
  def forward(self, x):
    """ x: (batch, length, chan, 3) """
    batch, length, chan, must_be[3] = x.shape
    x = x.permute(0, 3, 2, 1).reshape(3*batch, chan, length) # (batch*3, chan, length)
    y = self.conv(x)
    must_be[batch*3], must_be[chan], newlength = y.shape # length might have changed!
    y = y.reshape(batch, 3, chan, newlength).permute(0, 3, 2, 1) # (batch, newlength, chan, 3)
    return y


class EdgeRelativeEmbedMLPPath(nn.Module):
  """ input embedding for edges, where 2 positions are passed as input.
      assumes graph structure is path """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_v = VecLinear(4, vdim)
    self.scalar_layers = nn.Sequential(
      nn.Linear(4, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.conv_v = VecConv1d(vdim, 4, True)
    self.conv_a = ScalConv1d(adim, 4, True)
  def forward(self, pos_0, pos_1):
    """ pos0: (batch, nodes, 3)
        pos1: (batch, nodes, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, nodes, adim)
        v_out: (batch, nodes, vdim, 3) """
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    vecs = torch.stack([ # 4 relative vectors
        pos_0[:,  1:] - pos_0[:, :-1],
        pos_1[:,  1:] - pos_1[:, :-1],
        pos_1[:, :-1] - pos_0[:,  1:],
        pos_1[:,  1:] - pos_0[:, :-1],
      ], dim=2) # (batch, nodes - 1, 4, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, nodes - 1, 4)
    y_a = self.scalar_layers(norms)
    y_v = self.lin_v(vecs)
    a_out = self.conv_a(y_a)
    v_out = self.conv_v(y_v)
    return a_out, v_out


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
    self.layers_v = nn.Sequential(
      VecLinear(vdim, vdim),
      VecRootS(),
      VecLinear(vdim, vdim),
      VecRootS(),
      VecLinear(vdim, vdim))
    self.av_prod = ScalVecProducts(adim, vdim, rank)
    self.gnorm_a = ScalGroupNorm(adim, agroups)
    self.gnorm_v = VecGroupNorm(vdim, vgroups)
  def forward(self, x_a, x_v):
    y_a = self.layers_a(x_a)
    y_v = self.layers_v(x_v)
    p_a, p_v = self.av_prod(x_a, x_v)
    z_a, z_v = self.gnorm_a(y_a + p_a), self.gnorm_v(y_v + p_v)
    return (x_a + z_a), (x_v + z_v)


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    agroups, vgroups = config["agroups"], config["vgroups"]
    self.edge_embed = EdgeRelativeEmbedMLPPath(adim, vdim)
    self.node_embed = NodeRelativeEmbedMLP(adim, vdim)
    self.conv_0_a = ScalConv1d(adim, 7)
    self.conv_0_v = VecConv1d(vdim, 7)
    self.local_res = LocalResidual(config)
    self.conv_1_a = ScalConv1d(adim, 7)
    self.conv_1_v = VecConv1d(vdim, 7)
    self.gnorm_a = ScalGroupNorm(adim, agroups)
    self.gnorm_v = VecGroupNorm(vdim, vgroups)
  def get_embedding(self, pos_0, pos_1):
    emb_edge_a, emb_edge_v = self.edge_embed(pos_0, pos_1)
    emb_node_a, emb_node_v = self.node_embed(pos_0, pos_1)
    return emb_edge_a + emb_node_a, emb_edge_v + emb_node_v
  def forward(self, tup):
    pos_0, pos_1, x_a, x_v = tup
    emb_a, emb_v = self.get_embedding(pos_0, pos_1)
    y_a = self.conv_0_a(x_a) + emb_a
    y_v = self.conv_0_v(x_v) + emb_v
    y_a, y_v = self.local_res(y_a, y_v)
    z_a = self.gnorm_a(self.conv_1_a(y_a))
    z_v = self.gnorm_v(self.conv_1_v(y_v))
    return pos_0, pos_1, (x_a + z_a), (x_v + z_v)



class Discriminator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    self.lin_a = nn.Linear(adim, 1, bias=False)
    self.lin_v = nn.Linear(vdim, 1, bias=False)
  def forward(self, pos_0, pos_1):
    x_a, x_v = self.node_enc(pos_0, pos_1)
    *_, y_a, y_v = self.blocks((pos_0, pos_1, x_a, x_v))
    v_norms = torch.linalg.vector_norm(y_v, dim=-1)
    return (self.lin_a(y_a) + self.lin_v(v_norms)).sum(2).mean(1)


class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks1 = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    self.blocks2 = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    self.blocks_tune = nn.Sequential(
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.lin_v_node1 = VecLinear(vdim, 1)
    self.lin_v_node2 = VecLinear(vdim, 1)
    self.lin_v_node_tune = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def _predict1(self, pos0, noised):
    x_a, x_v = self.node_enc(pos0, noised)
    *_, y_a, y_v = self.blocks1((pos0, noised, x_a, x_v))
    return self.lin_v_node1(y_v)[:, :, 0]*self.out_norm_coeff, y_a, y_v
  def _predict2(self, pos0, noised, x_a, x_v):
    *_, y_a, y_v = self.blocks2((pos0, noised, x_a, x_v))
    return self.lin_v_node2(y_v)[:, :, 0]*self.out_norm_coeff, y_a, y_v
  def _finetune(self, pos0, noised, x_a, x_v, ε_a, ε_v):
    x_a = x_a + ε_a
    x_v = x_v + ε_v
    *_, y_v = self.blocks_tune((pos0, noised, x_a, x_v))
    return self.lin_v_node_tune(y_v)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, noise_1, noise_2, ε_a, ε_v):
    noised = pos0 + noise_1
    pred_noise, x_a, x_v = self._predict1(pos0, noised)
    noised = noised - pred_noise
    noised = noised + noise_2
    pred_noise, x_a, x_v = self._predict2(pos0, noised, x_a, x_v)
    noised = noised - pred_noise
    return noised + self._finetune(pos0, noised, x_a, x_v, ε_a, ε_v)



class WGAN3D:
  is_gan = True
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    assert config.sim.space_dim == 3
    assert config.cond_type == Condition.COORDS
    assert config.subtract_mean == 0
    self.n_nodes = config.sim.poly_len
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.AdamW(self.disc.parameters(), self.config["lr_d"], betas, weight_decay=self.config["weight_decay"])
    self.optim_g = torch.optim.AdamW(self.gen.parameters(),  self.config["lr_g"], betas, weight_decay=self.config["weight_decay"])
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
    data, cond = data.reshape(-1, self.n_nodes, 3), cond.reshape(-1, self.n_nodes, 3)
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, data, cond):
    self.optim_d.zero_grad()
    # train on real data (with instance noise)
    instance_noise_r = self.config["inst_noise_str_r"]*torch.randn_like(data)
    r_data = data + instance_noise_r # instance noise
    y_r = self.disc(cond, r_data)
    # train on generated data (with instance noise)
    g_data = self.generate(cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.disc(cond, g_data)
    # endpoint penalty on interpolated data
    mix_factors1 = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    mixed_data1 = mix_factors1*g_data + (1 - mix_factors1)*r_data
    y_mixed1 = self.disc(cond, mixed_data1)
    mix_factors2 = torch.rand(cond.shape[0], 1, 1, device=self.config.device)
    mixed_data2 = mix_factors2*g_data + (1 - mix_factors2)*r_data
    y_mixed2 = self.disc(cond, mixed_data2)
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
    y_g = self.disc(cond, g_data)
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
    batch, must_be[self.n_nodes], must_be[3] = cond.shape
    pos_noise_1, pos_noise_2, z_a, z_v = self.get_latents(batch)
    return self.gen(cond, pos_noise_1, pos_noise_2, z_a, z_v)
  def get_latents(self, batchsz):
    """ sample latent space for generator """
    pos_noise_1 = self.config["z_scale_1"]*torch.randn(batchsz, self.n_nodes, 3, device=self.config.device)
    pos_noise_2 = self.config["z_scale_2"]*torch.randn(batchsz, self.n_nodes, 3, device=self.config.device)
    z_a = torch.randn(batchsz, self.n_nodes, self.config["adim"], device=self.config.device)
    z_v = torch.randn(batchsz, self.n_nodes, self.config["vdim"], 3, device=self.config.device)
    return pos_noise_1, pos_noise_2, z_a, z_v
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()
  def predict(self, cond):
    batch, must_be[3*self.n_nodes] = cond.shape
    with torch.no_grad():
      return self.generate(cond.reshape(batch, self.n_nodes, 3)).reshape(batch, 3*self.n_nodes)



# export model class and trainer class:
modelclass   = WGAN3D
trainerclass = GANTrainer



