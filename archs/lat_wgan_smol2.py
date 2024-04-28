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

nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 128       # Size of feature maps in generator
ndf = 128       # Size of feature maps in discriminator


class EncCondition(nn.Module):
  def __init__(self, config):
    super(EncCondition, self).__init__()
    self.lin_inp = nn.Linear(config.cond_dim, 2*ndf)
    self.lin_cnd = nn.Linear(config.cond_dim, 2*ndf)
    self.layers_middle = nn.Sequential(
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(2*ndf, ndf)
    )
    self.lin_l = nn.Linear(ndf, ndf)
    self.lin_r = nn.Linear(ndf, ndf)
  def forward(self, inp, cnd):
    enc_inp = self.lin_inp(inp)
    enc_cnd = self.lin_cnd(cnd)
    y = self.layers_middle(enc_inp + enc_cnd) 
    return self.lin_l(y)*self.lin_r(y)


class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()
    cond_dim = config.cond_dim
    self.layers_0 = nn.Linear(cond_dim, ndf)
    self.enc_1 = EncCondition(config)
    self.layers_1 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.enc_2 = EncCondition(config)
    self.layers_2 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.enc_3 = EncCondition(config)
    self.layers_3 = nn.Sequential(
      nn.Linear(ndf, ndf),
      nn.BatchNorm1d(ndf),
      nn.LeakyReLU(0.2, inplace=True))
    self.head = nn.Linear(ndf, 1)
  def forward(self, inp, cond):
    y1 = self.layers_0(inp - cond)
    y2 = self.layers_1(y1) + self.enc_1(inp, cond)
    y3 = self.layers_2(y2) + self.enc_2(inp, cond)
    y4 = self.layers_3(y3) + self.enc_3(inp, cond)
    return self.head(y4)


class Generator(nn.Module):
  """ a small generator... """
  def __init__(self, config):
    super(Generator, self).__init__()
    cond_dim = config.cond_dim
    self.lin_z1 = nn.Linear(nz, ngf)
    self.lin_z2 = nn.Linear(nz, ngf)
    self.lin_c = nn.Linear(cond_dim, ngf)
    self.activ = nn.LeakyReLU(0.2, inplace=True)
    self.lin_out = nn.Linear(ngf, cond_dim)
  def forward(self, input, cond):
    z1, z2 = input
    y = self.lin_z1(z1) + self.lin_c(cond)
    y_act = self.activ(y) + self.lin_z2(z2)
    return self.lin_out(y_act)



# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


class LatentGANTrainer:
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
    return LatentGANTrainer(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.apply(weights_init)
    gen.apply(weights_init)
    return LatentGANTrainer(disc, gen, config)
  def save_to_dict(self):
    return {
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    loss_d = self.disc_step(data, cond)
    loss_g = self.gen_step(cond)
    return loss_d, loss_g
  def disc_step(self, r_data, cond):
    self.optim_d.zero_grad()
    # train on real data
    y_r = self.disc(r_data, cond)
    # train on generated data
    g_data = self.gen(self.get_latents(cond.shape[0]), cond)
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
      torch.randn(batchsz, nz, device=self.config.device))
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()



# export model class:
modelclass = LatentGANTrainer



