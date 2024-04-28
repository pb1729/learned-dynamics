import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from config import Config, Condition


# This is the architecture source file for training a VAE on the distribution of system states.

# constants:
lr = 0.0008     # Learning rate
beta_1 = 0.5    # Adam parameter
beta_2 = 0.99   # Adam parameter
nf = 128        # number of features in nets
w_rec = 800.    # reconstruction weight


def get_input_preproc_layers(config):
  """ preprocess input by subtracting mean if necessary """
  layer_list = []
  if config.cond_type == Condition.COORDS and config.subtract_mean:
    if config.x_only:
      layer_list.append(SubtractMean(config.state_dim, config.subtract_mean))
    else:
      layer_list.append(SubtractMeanPos(config.state_dim, config.subtract_mean))
  return layer_list

class SubtractMean(nn.Module):
  """ given position coordinates as input, subtracts the mean """
  def __init__(self, state_dim, space_dim=1):
    super(SubtractMean, self).__init__()
    self.x_sz = state_dim
    self.space_dim = space_dim # number of spatial dimensions
    assert self.x_sz % self.space_dim == 0
  def forward(self, x):
    batch, state_dim = x.shape
    x = x.reshape(batch, -1, self.space_dim)
    x = x - x.mean(1, keepdim=True)
    return x.reshape(batch, self.x_sz)

class SubtractMeanPos(nn.Module):
  """ given position and velocity coordinates as input,
      subtracts the mean from the position coordinates only """
  def __init__(self, state_dim, space_dim=1):
    super(SubtractMeanPos, self).__init__()
    self.x_sz = state_dim // 2
    self.space_dim = space_dim # number of spatial dimensions
    assert self.x_sz % self.space_dim == 0
  def forward(self, xv):
    batch, state_dim = xv.shape
    x = xv[:, :self.x_sz].reshape(batch, -1, self.space_dim)
    x = x - x.mean(1, keepdim=True)
    return torch.cat([
      x.reshape(batch, self.x_sz),
      xv[:, self.x_sz:]], dim=1)


class Encoder(nn.Module):
  def __init__(self, config):
    super(Encoder, self).__init__()
    self.layers = nn.Sequential(
      *get_input_preproc_layers(config),
      nn.Linear(config.state_dim, nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.BatchNorm1d(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.BatchNorm1d(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.BatchNorm1d(nf),
      nn.LeakyReLU(0.2, inplace=True))
    self.mu_head = nn.Sequential(
      nn.Linear(nf, nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, config["nz"], bias=False),
    )
    self.log_sigma_head = nn.Sequential(
      nn.Linear(nf, nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, config["nz"]),
    )
  def forward(self, state):
    """ encoder forward pass. returns mu, sigma for latent vector z """
    y = self.layers(state)
    mu = self.mu_head(y)
    sigma = torch.exp(self.log_sigma_head(y))
    return mu, sigma


class Decoder(nn.Module):
  def __init__(self, config):
    super(Decoder, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(config["nz"], nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.BatchNorm1d(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.BatchNorm1d(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.BatchNorm1d(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, config.state_dim, bias=False))
  def forward(self, z):
    """ decoder forward pass. returns estimated mean value of state """
    return self.layers(z)



def weights_init(m):
  """ custom weights initialization """
  classname = m.__class__.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)



class Autoencoder:
  def __init__(self, enc, dec, config):
    self.enc = enc
    self.dec = dec
    self.config = config
    self.init_optim()
  def init_optim(self):
    parameters = itertools.chain(self.enc.parameters(), self.dec.parameters())
    self.optim = torch.optim.Adam(parameters, lr, (beta_1, beta_2))
  @staticmethod
  def load_from_dict(states, config):
    dec, enc = Decoder(config).to(config.device), Encoder(config).to(config.device)
    dec.load_state_dict(states["dec"])
    enc.load_state_dict(states["enc"])
    return Autoencoder(enc, dec, config)
  @staticmethod
  def makenew(config):
    dec, enc = Decoder(config).to(config.device), Encoder(config).to(config.device)
    enc.apply(weights_init)
    dec.apply(weights_init)
    return Autoencoder(enc, dec, config)
  def save_to_dict(self):
    return {
        "enc": self.enc.state_dict(),
        "dec": self.dec.state_dict(),
      }
  def train_step(self, data):
    self.optim.zero_grad()
    z_mu, z_sigma = self.enc(data)
    z = z_mu + z_sigma*torch.randn_like(z_sigma)
    data_rec = self.dec(z)
    loss = (w_rec*((data - data_rec)**2).sum(1) + 0.5*(z**2).sum(1) - 0.5*torch.log(z_sigma).sum(1)).mean(0)
    loss.backward()
    self.optim.step()
    return loss.item()
  def encode(self, state):
    with torch.no_grad():
      z_mu, z_sigma = self.enc(state)
      z = z_mu + z_sigma*torch.randn_like(z_sigma)
    return z
  def decode(self, z):
    with torch.no_grad():
      return self.dec(z)


modelclass = Autoencoder



