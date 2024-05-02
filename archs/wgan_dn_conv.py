import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Condition


# BASE SOURCE CODE FOR CONDITIONAL WGAN
#  * Uses CIFAR-10
#  * Conditions on average color of the image


lr_d = 0.0008   # learning rate for discriminator
lr_g = 0.0003   # learning rate for generator
beta_1 = 0.5    # Adam parameter
beta_2 = 0.99   # Adam parameter
k_L = 1.        # Lipschitz constant
in_str_g = 0.2  # Strength of Instance-Noise on generated data
in_str_r = 0.3  # Strength of Instance-Noise on real data

nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 128       # Size of feature maps in generator
ndf = 128       # Size of feature maps in discriminator


class ResidualConv1d(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv1d(dim, dim, 5, padding="same"),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(dim),
        nn.Conv1d(dim, dim, 5, padding="same"),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(dim),
      )
  def forward(self, x):
    return x + self.layers(x)


class ToAtomCoords(nn.Module):
  def __init__(self, space_dim):
    super().__init__()
    self.space_dim = space_dim
  def forward(self, x):
    """ x: (batch, n_atoms*space_dim) """
    batch, state_dim = x.shape
    assert state_dim % self.space_dim == 0
    n_atoms = state_dim // self.space_dim
    y = x.reshape(batch, n_atoms, self.space_dim)
    return y.transpose(1, 2)

class FromAtomCoords(nn.Module):
  def __init__(self, space_dim):
    super().__init__()
    self.space_dim = space_dim
  def forward(self, x):
    """ x: (batch, space_dim, n_atoms) """
    batch, space_dim, n_atoms = x.shape
    assert space_dim == self.space_dim
    return x.transpose(2, 1).reshape(batch, n_atoms*space_dim)

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
  def __init__(self, config):
    super().__init__()
    self.enc0 = StateEnc(1, ndf)
    self.enc1 = StateEnc(1, ndf)
    self.lin_pre = nn.Conv1d(ndf, ndf, 3, padding="same")
    self.layers = nn.Sequential(
      ResidualConv1d(ndf),
      ResidualConv1d(ndf),
      ResidualConv1d(ndf),
      nn.BatchNorm1d(ndf))
  def forward(self, tup):
    x0, x1, nn_vec = tup
    y = self.lin_pre(nn_vec) + self.enc0(x0) + self.enc1(x1)
    return x0, x1, self.layers(y)

class DiscHead(nn.Module):
  """ accepts tuple containing: (x0, x1, nn_vec) where x0, x1 are encoded states
      outputs a scalar """
  def __init__(self, config):
    super().__init__()
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
    state_dim, cond_dim = config.state_dim, config.cond_dim
    assert config.sim.space_dim == 1 # make net arch equivariant before allowing larger numbers, also unhardcode the 1's below
    assert config.cond_type == Condition.COORDS
    self.to_atom_coords = ToAtomCoords(1)
    self.enc_delta = StateEnc(1, ndf)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config),
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
    self.lin_out = nn.Conv1d(ndf, 1, 5, padding="same")
  def forward(self, tup):
    x0, x1, nn_vec = tup
    return self.lin_out(nn_vec)

class Generator(nn.Module):
  def __init__(self, config):
    super(Generator, self).__init__()
    state_dim, cond_dim = config.state_dim, config.cond_dim
    assert config.sim.space_dim == 1
    assert config.cond_type == Condition.COORDS
    self.to_atom_coords = ToAtomCoords(1)
    self.from_atom_coords = FromAtomCoords(1)
    self.enc_delta = StateEnc(1, ndf)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config),
      GenHead(config))
  def forward(self, input, cond):
    z_vec, noised = input
    x0, x1 = self.to_atom_coords(cond), self.to_atom_coords(noised)
    tup = (x0, x1, self.enc_delta(x1 - x0))
    return noised - self.from_atom_coords(self.blocks(tup))


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class GANTrainer:
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    self.init_optim()
  def init_optim(self):
    self.optim_d = torch.optim.Adam(self.disc.parameters(), lr_d, (beta_1, beta_2))
    self.optim_g = torch.optim.Adam(self.gen.parameters(),  lr_g, (beta_1, beta_2))
  @staticmethod
  def load_from_dict(states, config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return GANTrainer(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return GANTrainer(disc, gen, config)
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
    r_data = data + in_str_r*torch.randn_like(data) # instance noise
    y_r = self.disc(r_data, cond)
    # train on generated data (with instance noise)
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
    g_data = g_data + in_str_g*torch.randn_like(g_data) # instance noise
    y_g = self.disc(g_data, cond)
    # sample-delta penalty on interpolated data
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
    g_data = g_data + in_str_g*torch.randn_like(g_data) # instance noise
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
    return (
      torch.randn(batchsz, nz, device=self.config.device),
      20.*torch.randn(batchsz, self.config.state_dim, device=self.config.device))
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()



# export model class:
modelclass = GANTrainer



