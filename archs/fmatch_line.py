import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batched_model_eval
from config import Config, Condition
from layers_common import *
from attention_layers import *
from vamp_score import vamp_score


#
# Feature matching GAN based on the VAMP score
# Lives in the 1d space of lineland
#

class VAMPNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.x_only, "still need to implement having absolute velocity, so can't do x_only=False yet"
    nf, outdim = config["nf"], config["outdim"]
    self.layers = nn.Sequential(
      nn.Linear(config.state_dim, nf),
      Residual(nf),
      Residual(nf),
      Residual(nf),
      nn.Linear(nf, outdim),
    )
  def forward(self, x):
    return self.layers(x)

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
      Residual(ngf),
      Residual(ngf))
  def forward(self, noise, x0):
    inp = torch.cat([noise, x0], dim=-1)
    return x0 + noise[:, 0::15] + self.readout(self.layers2(
        self.input_embed2(inp) + self.layers1(
            self.input_embed1(inp)
        )
    ))


class KoopmanFMatch:
  """ class containing a VAMPNet and a generator trained by feature matching """
  is_gan = True
  def __init__(self, model, gen, config):
    self.model = model
    self.gen = gen
    self.config = config
    self.init_optim()
  def init_optim(self):
    self.optim = torch.optim.AdamW(self.model.parameters(),
      self.config["lr"], (self.config["beta_1"], self.config["beta_2"]), weight_decay=self.config["wd"])
    self.optim_g = torch.optim.AdamW(self.gen.parameters(),
      self.config["lr"], (self.config["beta_1"], self.config["beta_2"]))
  @staticmethod
  def load_from_dict(states, config):
    model = VAMPNet(config).to(config.device)
    gen = Generator(config).to(config.device)
    model.load_state_dict(states["vampnet"])
    gen.load_state_dict(states["gen"])
    return KoopmanFMatch(model, gen, config)
  @staticmethod
  def makenew(config):
    model = VAMPNet(config).to(config.device)
    gen = Generator(config).to(config.device)
    model.apply(weights_init)
    gen.apply(weights_init)
    return KoopmanFMatch(model, gen, config)
  def save_to_dict(self):
    return {
        "vampnet": self.model.state_dict(),
        "gen": self.gen.state_dict(),
      }
  def train_step(self, trajs):
    batch, L, must_be[self.config.state_dim] = trajs.shape
    # VAMPNet training step
    chi = self.model(trajs.reshape(batch*L, self.config.state_dim))
    chi = chi.reshape(batch, L, self.config["outdim"])
    trans_0, trans_1, C_10 = vamp_score(
      chi[:, :-1].reshape(batch*(L - 1), -1), chi[:, 1:].reshape(batch*(L - 1), -1),
      mode="all")
    loss = -(C_10**2).sum() # negative VAMP-2 score
    # Generator training step
    x_pred = self.generate(trajs[:, :-1].reshape(batch*(L - 1), self.config.state_dim))
    chi_pred = self.model(x_pred)
    chi_pred = trans_1.detach()(chi_pred)
    if self.config["by_matrix"]:
      chi = trans_0.detach()(chi.detach()[:, :-1].reshape(batch*(L - 1), -1))
      chi_1 = (C_10.detach() @ chi.T).T
    else:
      chi = trans_1.detach()(chi.detach()[:, 1:].reshape(batch*(L - 1), -1))
      chi_1 = chi
    loss_g = ((chi_pred - chi_1)**2).mean() # MSE loss
    # backprop and update
    self.optim.zero_grad()
    self.optim_g.zero_grad()
    (loss + loss_g).backward()
    self.optim.step()
    self.optim_g.step()
    return loss.item(), loss_g.item()
  def generate(self, cond):
    batch, must_be[self.config.state_dim] = cond.shape
    noise = self.get_latents(batch)
    return self.gen(noise, cond)
  def get_latents(self, batchsz):
    """ sample latents for generator """
    return torch.randn(batchsz, 15*self.config.cond_dim, device=self.config.device)
  def set_eval(self, bool_eval):
    if bool_eval:
      self.model.eval()
      self.gen.eval()
    else:
      self.model.train()
      self.gen.train()
  def predict(self, cond):
    with torch.no_grad():
      return self.generate(cond)


class KoopmanTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    loss, loss_g = self.model.train_step(trajs)
    print(f"{i}\t ℒ = {loss:05.6f} \t ℒᴳ = {loss_g:05.6f}")
    self.board.scalar("loss", i, loss)
    self.board.scalar("loss_g", i, loss_g)



# export model class and trainer class:
modelclass   = KoopmanFMatch
trainerclass = KoopmanTrainer




