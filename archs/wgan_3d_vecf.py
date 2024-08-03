import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Condition
from utils import must_be
from gan_common import GANTrainer
from layers_common import *
from attention_layers import *
from polymer_util import RouseEvolver



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
    self.probe_pts = ProbePoints(2, 2, adim, vdim)
    self.prox_attn = ProximityAttention(config["r0_list"], config["kq_dim"], (adim, vdim))
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
    probes = self.probe_pts(y_a, y_v, [pos_0, pos_1])
    probes_k, probes_q = probes[0], probes[1]
    dy_a, dy_v = self.prox_attn((y_a, y_v), probes_k, probes_q)
    #print(torch.sqrt((dy_a.detach()**2).mean()).item(), torch.sqrt((y_a.detach()**2).mean()).item()) # TODO
    y_a, y_v = y_a + dy_a, y_v + dy_v
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
      Block(config))
    self.lin_v = VecLinear(vdim, 1)
  def forward(self, pos_0, pos_1):
    x_a, x_v = self.node_enc(pos_0, pos_1)
    *_, y_a, y_v = self.blocks((pos_0, pos_1, x_a, x_v))
    # y_a not used, very sad! TODO?
    return self.lin_v(y_v)[:, :, 0]


class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.rouse_noiser = RouseEvolver(config)
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    self.blocks_tune = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.lin_v_node = VecLinear(vdim, 1)
    self.lin_v_node_tune = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def _predict(self, pos0, noised):
    x_a, x_v = self.node_enc(pos0, noised)
    *_, y_v = self.blocks((pos0, noised, x_a, x_v))
    return self.lin_v_node(y_v)[:, :, 0]*self.out_norm_coeff
  def _finetune(self, pos0, noised, ε_a, ε_v):
    x_a, x_v = self.node_enc(pos0, noised)
    x_a = x_a + ε_a
    x_v = x_v + ε_v
    *_, y_v = self.blocks_tune((pos0, noised, x_a, x_v))
    return self.lin_v_node_tune(y_v)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, noise, ε_a, ε_v):
    noised = self.rouse_noiser.predict(pos0, noise)
    pred_noise = self._predict(pos0, noised)
    noised = noised - pred_noise
    return noised + self._finetune(pos0, noised, ε_a, ε_v)



class WGAN3D:
  is_gan = True
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    assert config.sim.space_dim == 3
    assert config.cond_type == Condition.COORDS
    assert config.subtract_mean == 0
    assert config.x_only
    self.n_nodes = config.sim.poly_len
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.AdamW(self.disc.parameters(), self.config["lr_d"], betas, weight_decay=self.config["weight_decay"])
    self.optim_g = torch.optim.AdamW(self.gen.parameters(),  self.config["lr_g"], betas, weight_decay=self.config["weight_decay"])
    self.step_count = 0
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
  def sample_flow(self, x, x_rg):
    """ sample flow vector field function
        x, x_rg: (batch, nodes, 3) """
    Δx = x - x_rg
    #sq_dist = (torch.sqrt((Δx**2).sum(2, keepdim=True)).mean(1, keepdim=True))**2
    sq_dist = (Δx**2).sum(2, keepdim=True).mean(1, keepdim=True)
    return Δx/torch.sqrt(sq_dist + 1)
    #return Δx/(sq_dist**3 + 1.)
  def disc_step(self, r_data, cond):
    self.optim_d.zero_grad()
    g_data = self.generate(cond)
    x_data = self.generate(cond) + self.config["z_scale"]*torch.randn_like(g_data) # we'd like the discriminator flow field to be accurate on the space of generated data
    flow_frg = self.sample_flow(x_data, r_data) - self.sample_flow(x_data, g_data) # get a fragment of the actual flow
    flow_hat = self.disc(cond, x_data) # get predicted flow
    loss = ((flow_hat - flow_frg)**2).sum(2).mean() # average over batch and nodes
    loss.backward()
    self.optim_d.step()
    return loss.item()
  def gen_step(self, cond):
    self.optim_g.zero_grad()
    g_data = self.generate(cond)
    with torch.no_grad():
      flow = self.disc(cond, g_data)
    g_data.backward(gradient=flow) # ooh, not using a scalar loss function, spooky
    self.optim_g.step()
    return (flow**2).mean().item()
  def generate(self, cond):
    batch, must_be[self.n_nodes], must_be[3] = cond.shape
    pos_noise, z_a, z_v = self.get_latents(batch)
    return self.gen(cond, pos_noise, z_a, z_v)
  def get_latents(self, batchsz):
    """ sample latent space for generator """
    self.n_nodes
    pos_noise = torch.randn(batchsz, self.n_nodes, 3, device=self.config.device)
    z_a = torch.randn(batchsz, self.n_nodes, self.config["adim"], device=self.config.device)
    z_v = torch.randn(batchsz, self.n_nodes, self.config["vdim"], 3, device=self.config.device)
    return pos_noise, z_a, z_v
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




