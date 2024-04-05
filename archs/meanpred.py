import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Condition


# This is the architecture source file for training a simple mean predictor
# Architecture style is ResNet


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


class Residual(nn.Module):
  def __init__(self, dim):
    super(Residual, self).__init__()
    self.layers = nn.Sequential(
        nn.Linear(dim, dim),
        nn.BatchNorm1d(dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(dim, dim),
        nn.BatchNorm1d(dim),
        nn.LeakyReLU(0.2, inplace=True),
      )
    self.end_map = nn.Linear(dim, dim, bias=False)
  def forward(self, x):
    return self.end_map(self.layers(x))
  def self_init(self):
    nn.init.constant_(self.end_map.weight.data, 0.0)


class Meanpred(nn.Module):
  def __init__(self, config):
    super(Meanpred, self).__init__()
    nf = config["nf"]
    self.layers = nn.Sequential(
      *get_input_preproc_layers(config),
      nn.Linear(config.state_dim, nf),
      Residual(nf),
      Residual(nf),
      Residual(nf),
      Residual(nf),
      Residual(nf),
      nn.Linear(nf, config.state_dim, bias=False),
    )
  def forward(self, z):
    """ decoder forward pass. returns estimated mean value of state """
    return self.layers(z)



def weights_init(m):
  """ custom weights initialization """
  cls = m.__class__
  if hasattr(cls, "self_init"):
    m.self_init()
  classname = cls.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)



class MeanpredTrainer:
  def __init__(self, meanpred, config):
    self.meanpred = meanpred
    self.config = config
    self.init_optim()
  def init_optim(self):
    self.optim = torch.optim.Adam(self.meanpred.parameters(), self.config["lr"], (self.config["beta_1"], self.config["beta_2"]))
  @staticmethod
  def load_from_dict(states, config):
    meanpred = Meanpred(config).to(config.device)
    meanpred.load_state_dict(states["meanpred"])
    return MeanpredTrainer(meanpred, config)
  @staticmethod
  def makenew(config):
    meanpred = Meanpred(config).to(config.device)
    meanpred.apply(weights_init)
    return MeanpredTrainer(meanpred, config)
  def save_to_dict(self):
    return {
        "meanpred": self.meanpred.state_dict(),
      }
  def train_step(self, data, cond):
    self.optim.zero_grad()
    y_hat = self.meanpred(cond)
    loss = ((data - y_hat)**2).sum(1).mean(0)
    loss.backward()
    self.optim.step()
    return loss.item()
  def predict(self, cond):
    with torch.no_grad():
      y_hat = self.meanpred(cond)
    return y_hat
  def set_eval(self, cond):
    if cond:
      self.meanpred.eval()
    else:
      raise NotImplementedError("Setting mode to un-evaluation is not implemented.")


modelclass = MeanpredTrainer



