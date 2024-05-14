import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Condition
from gan_common import GANTrainer
from layers_common import weights_init, ResidualConv1d, ToAtomCoords, FromAtomCoords


# BASE SOURCE CODE FOR CONDITIONAL WGAN
#  * Uses CIFAR-10
#  * Conditions on average color of the image

k_L = 1.        # Lipschitz constant




class StateEnc(nn.Module):
  """ encode a state """
  def __init__(self, space_dim, chan):
    super().__init__()
    self.layers = nn.Conv1d(space_dim, chan, 5, padding="same")
  def forward(self, state):
    return self.layers(state)

class Block(nn.Module):
  """ accepts tuple containing: (x0, x1, nn_vec) where x0, x1 are encoded states
      outputs same kind of tuple """
  def __init__(self, df):
    super().__init__()
    self.enc0 = StateEnc(1, df)
    self.enc1 = StateEnc(1, df)
    self.lin_pre = nn.Conv1d(df, df, 3, padding="same")
    self.layers = nn.Sequential(
      ResidualConv1d(df),
      ResidualConv1d(df),
      ResidualConv1d(df),
      nn.BatchNorm1d(df))
  def forward(self, tup):
    x0, x1, nn_vec = tup
    y = self.lin_pre(nn_vec) + self.enc0(x0) + self.enc1(x1)
    return x0, x1, self.layers(y)

class DiscHead(nn.Module):
  """ accepts tuple containing: (x0, x1, nn_vec) where x0, x1 are encoded states
      outputs a scalar """
  def __init__(self, config):
    super().__init__()
    ndf = config["ndf"]
    self.lin_pre = nn.Conv1d(ndf, 2*ndf, 1)
    self.layers = nn.Sequential(
      nn.BatchNorm1d(2*ndf),
      nn.Linear(2*ndf, 2*ndf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(2*ndf, 2*ndf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(2*ndf, 1))
  def forward(self, tup):
    x0, x1, nn_vec = tup
    y = self.lin_pre(nn_vec)
    return self.layers(y.sum(2))

class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()
    ndf = config["ndf"]
    assert config.sim.space_dim == 1 # make net arch equivariant before allowing larger numbers, also unhardcode the 1's below
    assert config.cond_type == Condition.COORDS
    self.to_atom_coords = ToAtomCoords(1)
    self.enc_delta = StateEnc(1, ndf)
    self.blocks = nn.Sequential(
      Block(ndf),
      Block(ndf),
      Block(ndf),
      Block(ndf),
      DiscHead(config))
  def forward(self, inp, cond):
    x0, x1 = self.to_atom_coords(cond), self.to_atom_coords(inp)
    tup = (x0, x1, self.enc_delta(x1 - x0))
    return self.blocks(tup)

class GenHead(nn.Module):
  """ accepts tuple containing: (x0, x1, nn_vec) where x0, x1 are encoded states
      outputs an estimate of the noise (has same shape as a state) """
  def __init__(self, config):
    super().__init__()
    ngf = config["ngf"]
    self.lin_out = nn.Conv1d(ngf, 1, 5, padding="same")
  def forward(self, tup):
    x0, x1, nn_vec = tup
    return self.lin_out(nn_vec)

class Generator(nn.Module):
  def __init__(self, config):
    super(Generator, self).__init__()
    ngf = config["ngf"]
    assert config.sim.space_dim == 1
    assert config.cond_type == Condition.COORDS
    self.to_atom_coords = ToAtomCoords(1)
    self.from_atom_coords = FromAtomCoords(1)
    self.enc_delta = StateEnc(1, ngf)
    self.blocks = nn.Sequential(
      Block(ngf),
      Block(ngf),
      Block(ngf),
      Block(ngf),
      GenHead(config))
  def forward(self, noised, cond):
    x0, x1 = self.to_atom_coords(cond), self.to_atom_coords(noised)
    tup = (x0, x1, self.enc_delta(x1 - x0))
    return noised - self.from_atom_coords(self.blocks(tup))


class GAN:
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    self.init_optim()
  def init_optim(self):
    betas = (self.config["beta_1"], self.config["beta_2"])
    self.optim_d = torch.optim.Adam(self.disc.parameters(), self.config["lr_d"], betas)
    self.optim_g = torch.optim.Adam(self.gen.parameters(),  self.config["lr_g"], betas)
  @staticmethod
  def load_from_dict(states, config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return GAN(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return GAN(disc, gen, config)
  def save_to_dict(self):
    return {
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, data, cond):
    self.optim_d.zero_grad()
    # train on real data (with instance noise)
    instance_noise_r = self.config["inst_noise_str_r"]*torch.randn_like(data)
    r_data = data + instance_noise_r # instance noise
    y_r = self.disc(r_data, cond)
    # train on generated data (with instance noise)
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.disc(g_data, cond)
    # endpoint penalty on interpolated data
    mix_factors1 = torch.rand(cond.shape[0], 1, device=self.config.device)
    mixed_data1 = mix_factors1*g_data + (1 - mix_factors1)*r_data
    y_mixed1 = self.disc(mixed_data1, cond)
    mix_factors2 = torch.rand(cond.shape[0], 1, device=self.config.device)
    mixed_data2 = mix_factors2*g_data + (1 - mix_factors2)*r_data
    y_mixed2 = self.disc(mixed_data2, cond)
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
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    instance_noise_g = self.config["inst_noise_str_g"]*torch.randn_like(g_data)
    g_data = g_data + instance_noise_g # instance noise
    y_g = self.disc(g_data, cond)
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
  def get_latents(self, batchsz):
    """ sample latents for generator """
    return self.config["z_scale"]*torch.randn(batchsz, self.config.state_dim, device=self.config.device)
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()



# export model class and trainer class:
modelclass   = GAN
trainerclass = GANTrainer



