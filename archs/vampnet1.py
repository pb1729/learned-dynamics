import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batched_model_eval
from config import Config, Condition
from layers_common import weights_init, Residual, ResidualConv1d, ToAtomCoords
from vamp_score import vamp_score, Affine



def res_layers_weight_decay(model, coeff=0.1):
  """ weight decay only for weights in ResLayer's """
  params = []
  for m in model.modules():
    if isinstance(m, Residual) or isinstance(m, ResidualConv1d):
      for layer in m.layers:
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
          params.append(layer.weight)
          if layer.bias is not None: params.append(layer.bias)
  return coeff * sum((param**2).sum() for param in params)


class VAMPNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.x_only, "still need to implement having absolute velocity, so can't do x_only=False yet"
    assert config.sim.space_dim == 1
    nf, outdim = config["nf"], config["outdim"]
    self.conv_layers = nn.Sequential(
      ToAtomCoords(1),
      nn.Conv1d(1, nf, 5, padding="same"),
      ResidualConv1d(nf),
      ResidualConv1d(nf),
      nn.Conv1d(nf, nf, 5, padding="same"),
      ResidualConv1d(nf),
      ResidualConv1d(nf),
      nn.Conv1d(nf, 2*nf, 3, padding="same"),
    )
    self.flat_layers = nn.Sequential(
      nn.Linear(2*nf, 2*nf),
      Residual(2*nf),
      Residual(2*nf),
      Residual(2*nf),
      nn.Linear(2*nf, outdim),
    )
  def forward(self, z):
    """ decoder forward pass. returns estimated mean value of state """
    return self.flat_layers(self.conv_layers(z).sum(2))



class KoopmanModel:
  """ class containing a VAMPNet and the full set of associated transformations """
  def __init__(self, model, trans_0, trans_1, S, config):
    self.model = model
    self.trans_0 = trans_0
    self.trans_1 = trans_1
    self.S = S
    self.config = config
    self.init_optim()
  def init_optim(self):
    self.optim = torch.optim.Adam(self.model.parameters(),
      self.config["lr"], (self.config["beta_1"], self.config["beta_2"]))
  @staticmethod
  def load_from_dict(states, config):
    model = VAMPNet(config).to(config.device)
    model.load_state_dict(states["vampnet"])
    return KoopmanModel(model, states["trans_0"], states["trans_1"], states["S"], config)
  @staticmethod
  def makenew(config):
    model = VAMPNet(config).to(config.device)
    model.apply(weights_init)
    return KoopmanModel(model, None, None, None, config)
  def save_to_dict(self):
    return {
        "vampnet": self.model.state_dict(),
        "trans_0": self.trans_0,
        "trans_1": self.trans_1,
        "S": self.S,
      }
  def train_step(self, trajs):
    N, L, in_dim = trajs.shape
    self.model.zero_grad()
    chi = self.model(trajs.reshape(N*L, in_dim)).reshape(N, L, -1)
    chi_0 = chi[:, :-1].reshape(N*(L - 1), -1)
    chi_1 = chi[:, 1: ].reshape(N*(L - 1), -1)
    loss = -vamp_score(chi_0, chi_1)
    loss_wd = res_layers_weight_decay(self.model, coeff=self.config["wd"])
    total_loss = loss + loss_wd
    total_loss.backward()
    self.optim.step()
    return loss.item(), loss_wd.item()
  def calculate_transforms(self, dataset):
    N, L, in_dim = dataset.shape
    assert in_dim == self.config.state_dim
    with torch.no_grad():
      chi = batched_model_eval(self.model, dataset.reshape(N*L, in_dim), self.config["outdim"])
      chi_0 = chi.reshape(N, L, -1)[:, :-1].reshape(N*(L-1), -1)
      chi_1 = chi.reshape(N, L, -1)[:, 1: ].reshape(N*(L-1), -1)
      trans_0, trans_1, K = vamp_score(chi_0, chi_1, mode="all")
      trans_0, trans_1, K = trans_0.detach(), trans_1.detach(), K.detach()
    U, S, Vh = torch.linalg.svd(K)
    dim, = S.shape
    trans_0 = Affine( (Vh @ trans_0.W)[:dim], trans_0.mu[:dim])
    trans_1 = Affine((U.T @ trans_1.W)[:dim], trans_1.mu[:dim])
    self.trans_0, self.trans_1, self.S = trans_0, trans_1, S
  def eigenfn_0(self, data):
    """ compute eigenfunction 0 on data """
    return self.trans_0(batched_model_eval(self.model, data, self.config["outdim"]))
  def eigenfn_1(self, data):
    """ compute eigenfunction 1 on data """
    return self.trans_1(batched_model_eval(self.model, data, self.config["outdim"]))
  def eval_score(self, dataset):
    """ evaluate performance of this model on a test dataset """
    N, L, _ = dataset.shape
    with torch.no_grad():
      x = self.eigenfn_0(dataset[:, :-1].reshape(N*(L-1), -1))
      y = self.eigenfn_1(dataset[:, 1: ].reshape(N*(L-1), -1))
      mu_x = x.mean(0)
      mu_y = y.mean(0)
      var_x = (x**2).mean(0) - mu_x**2
      var_y = (y**2).mean(0) - mu_y**2
      corr = (x*y).mean(0) - mu_x*mu_y
      ratio = corr / torch.sqrt(var_x*var_y)
    return ratio
  def set_eval(self, bool_eval):
    if bool_eval:
      self.disc.eval()
      self.gen.eval()
    else:
      self.disc.train()
      self.gen.train()


class KoopmanTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
    self.final_tuning_data = [] # data stash saved for final tuning of model
  def step(self, i, trajs):
    loss, loss_wd = self.model.train_step(trajs)
    loss_tot = loss + loss_wd
    print(f"{i}\t ℒ = {loss:05.6f}\t ℒᵂᴰ = {loss_wd:05.6f}\tℒᵀᴼᵀ = {loss_tot:05.6f}")
    self.board.scalar("loss", i, loss)
    self.board.scalar("loss_wd", i, loss_wd)
    self.board.scalar("loss_tot", i, loss_tot)
    if i < self.model.config["tuning_batches"]:
      self.final_tuning_data.append(trajs)
    if i + 1 == self.model.config.nsteps:
      self.perform_final_tuning()
  def perform_final_tuning(self):
    print("performing final tuning of the model's transforms...")
    # combine tuning data into one big batch
    dataset = torch.cat(self.final_tuning_data, dim=0)
    self.model.calculate_transforms(dataset)
    print("tuning finished!")



# export model class and trainer class:
modelclass   = KoopmanModel
trainerclass = KoopmanTrainer




