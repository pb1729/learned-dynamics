import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Condition
from layers_common import ResidualConv1d, ToAtomCoords, FromAtomCoords


# This is the architecture source file for training a simple mean predictor
# Architecture style is Convolutional ResNet
# (convolutions are 1d and go along the length of the polymer)
# Predictions are of the relative change in position compared to current position


class Meanpred(nn.Module):
  def __init__(self, config):
    super(Meanpred, self).__init__()
    assert config.x_only, "still need to implement having absolute velocity, so can't do x_only=False yet"
    nf = config["nf"]
    assert config.sim.space_dim == 1
    self.layers = nn.Sequential(
      ToAtomCoords(1),
      nn.Conv1d(1, nf, 5, padding="same"),
      ResidualConv1d(nf),
      ResidualConv1d(nf),
      nn.Conv1d(nf, nf, 5, padding="same"),
      ResidualConv1d(nf),
      ResidualConv1d(nf),
      nn.Conv1d(nf, 1, 5, padding="same"),
      FromAtomCoords(1),
    )
  def forward(self, z):
    """ decoder forward pass. returns estimated mean value of state """
    return z + self.layers(z) # estimate is relative to current values



def weights_init(m):
  """ custom weights initialization """
  cls = m.__class__
  if hasattr(cls, "self_init"):
    m.self_init()
  classname = cls.__name__
  if classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)



class MeanpredModel:
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
    return MeanpredModel(meanpred, config)
  @staticmethod
  def makenew(config):
    meanpred = Meanpred(config).to(config.device)
    meanpred.apply(weights_init)
    return MeanpredModel(meanpred, config)
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

class MeanpredTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    N, L, state_dim = trajs.shape
    cond = self.model.config.cond(trajs[:, :-1].reshape(N*(L - 1), state_dim))
    data = trajs[:, 1:].reshape(N*(L - 1), state_dim)
    loss = self.model.train_step(data, cond)
    print(f"{i}\t ℒ = {loss:05.6f}")
    self.board.scalar("loss", i, loss)


# export model class and trainer class:
modelclass   = MeanpredModel
trainerclass = MeanpredTrainer



