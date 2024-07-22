import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Condition
from gan_common import GANTrainer
from optimizers import FriendlyAverage34
from utils import must_be

# TODO: for debugging, remove along with associated function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

from sims import get_poly_tc


# simple WGAN for a system with no particular properties
# (not decomposable into beads, no spatial symmetries)


class Residual(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(dim, dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.LayerNorm(dim),
        nn.Linear(dim, dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.LayerNorm(dim),
      )
  def forward(self, x):
    return x + self.layers(x)

class ResidualLin(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
          )
    def forward(self, x):
        return x + self.layers(x)

class Discriminator(nn.Module):
  def __init__(self, config):
    super().__init__()
    ndf = config["ndf"]
    dim = config.cond_dim
    self.input_embed1 = nn.Linear(2*dim, ndf)
    self.input_embed2 = nn.Linear(2*dim, ndf)
    self.readout = nn.Linear(ndf, 1)
    self.layers1 = nn.Sequential(
      Residual(ndf),
      Residual(ndf))
    self.layers2 = nn.Sequential(
      Residual(ndf),
      Residual(ndf))
  def forward(self, x0, x1):
    inp = torch.cat([x0, x1], dim=-1)
    return self.readout(self.layers2(
        self.input_embed2(inp) + self.layers1(
            self.input_embed1(inp)
        )
    ))

class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    ngf = config["ngf"]
    dim = config.cond_dim
    self.input_embed1 = nn.Linear(16*dim, ngf)
    self.input_embed2 = nn.Linear(16*dim, ngf)
    self.readout = nn.Linear(ngf, dim)
    self.layers1 = nn.Sequential(
      Residual(ngf),
      Residual(ngf))
    self.layers2 = nn.Sequential(
      ResidualLin(ngf),
      ResidualLin(ngf))
  def forward(self, noise, x0):
    inp = torch.cat([noise, x0], dim=-1)
    return x0 + noise[:, 0::15] + self.readout(self.layers2(
        self.input_embed2(inp) + self.layers1(
            self.input_embed1(inp)
        )
    ))



class GAN:
  is_gan = True
  def __init__(self, disc, gen, config):
    self.disc = disc
    self.gen  = gen
    self.config = config
    self.init_optim()
  def init_optim(self):
    disc_decay_params = (param for param in self.disc.parameters() if param.flatten()[0] != 1.)
    disc_nodecay_params = (param for param in self.disc.parameters() if param.flatten()[0] == 1.)
    self.optim_d = FriendlyAverage34(disc_decay_params, self.config["lr_d"], weight_decay=self.config["weight_decay"])
    self.optim_d.add_param_group({
      "params": disc_nodecay_params,
      "weight_decay": None})
    self.optim_g = FriendlyAverage34(self.gen.parameters(),  self.config["lr_g"])
    self.step_count = 0
    d_params = 0
    for param in self.disc.parameters():
      d_params += 1
    g_params = 0
    for param in self.gen.parameters():
      g_params += 1
    self.weight_histories_d = np.zeros((self.config.nsteps, d_params)) + np.nan
    self.weight_histories_g = np.zeros((self.config.nsteps, g_params)) + np.nan
  @staticmethod
  def load_from_dict(states, config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    disc.load_state_dict(states["disc"])
    gen.load_state_dict(states["gen"])
    return GAN(disc, gen, config)
  @staticmethod
  def makenew(config):
    disc, gen = Discriminator(config).to(config.device), Generator(config).to(config.device)
    return GAN(disc, gen, config)
  def save_to_dict(self):
    return {
        "disc": self.disc.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, data, cond):
    # training steps
    if self.step_count % 3 == 0:
      loss_g = self.gen_step(cond)
    else:
      loss_g = 0
    loss_d = self.disc_step(data, cond)
    # save parameters and other data, maybe record a frame
    if self.step_count % 3 == 0:
      self.record_frame(self.step_count//3)
    for j, param in enumerate(self.disc.parameters()):
      self.weight_histories_d[self.step_count, j] = param.reshape(-1)[0].detach().item()
    for j, param in enumerate(self.gen.parameters()):
      self.weight_histories_g[self.step_count, j] = param.reshape(-1)[0].detach().item()
    self.step_count += 1
    if self.step_count % 160 == 0:
      for group in self.optim_g.param_groups: # learning rate schedule
        group["lr"] *= 0.9
      for group in self.optim_d.param_groups: # learning rate schedule
        group["lr"] *= 0.975
    return loss_d, loss_g
  def disc_step(self, r_data, cond):
    self.optim_d.zero_grad()
    y_r = self.disc(r_data, cond)
    # train on generated data
    g_data = self.gen(self.get_latents(cond.shape[0]), cond) # note that we noise from 0 rather than starting from cond...
    y_g = self.disc(g_data, cond)
    loss = y_r - y_g
    loss = loss + torch.relu(abs(y_r - y_g)/torch.sqrt(((r_data - g_data)**2).sum(1) + 0.01) - 1.) # endpoint penalty
    loss = loss + 0.2*(abs(y_r - y_g)/torch.sqrt(((r_data - g_data)**2).sum(1) + 0.01))**2 # endpoint penalty, L2
    loss = loss.mean()
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
  def get_latents(self, batchsz):
    """ sample latents for generator """
    return torch.randn(batchsz, 15*self.config.cond_dim, device=self.config.device)
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()
  def predict(self, cond):
    batch = cond.shape[0]
    return self.gen(self.get_latents(batch), cond)
  def cond_sample(self, x0):
    tau = self.config.sim.delta_t/get_poly_tc(self.config.sim, 1.)
    return (2.718281828**(-tau))*x0 + torch.randn_like(x0)/((1. - 2.718281828**(-2*tau))**0.5)
  def record_frame(self, i, hist_datas=1000):
    with torch.no_grad():
      # initialize plot
      fig, (ax, ax_wts) = plt.subplots( nrows=2, ncols=1 )
      # plot hists of generated points
      x0 = torch.tensor(2., device=self.config.device).reshape(1,1)
      x1 = self.cond_sample(x0.expand(hist_datas, 1)).detach().cpu()
      x1_hat = self.gen(self.get_latents(hist_datas), x0.expand(hist_datas, 1)).detach().cpu()
      ax.hist(x1[:, 0], bins=20, alpha=0.4)
      ax.hist(x1_hat[:, 0], bins=20, alpha=0.4)
      # plot discriminator line
      x = torch.linspace(-8., 8., 600, device=self.config.device).reshape(600, 1)
      y = self.disc(x0.expand(600, 1), x).detach().cpu()
      ax_disc = ax.twinx()
      ax_disc.plot(x.detach().cpu(), y.detach().cpu(), color="green")
      # plot weight history:
      for j in range(self.weight_histories_d.shape[1]):
        ax_wts.plot(self.weight_histories_d[:, j])
      for j in range(self.weight_histories_g.shape[1]):
        ax_wts.plot(self.weight_histories_g[:, j])
      # save to file
      fig.savefig('slides/%d.png' % i)
      plt.clf() # https://stackoverflow.com/a/72890978
      plt.close(fig)





# export model class and trainer class:
modelclass   = GAN
trainerclass = GANTrainer



