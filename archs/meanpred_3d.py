import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn

from config import Condition
from utils import must_be
from layers_common import *


class ScalConv1d(nn.Module):
  def __init__(self, chan, kernsz, edges_to_nodes=False):
    super().__init__()
    if edges_to_nodes:
      assert kernsz % 2 == 0
      self.conv = nn.Conv1d(chan, chan, kernsz, padding=(kernsz//2))
    else:
      assert kernsz % 2 == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, padding="same")
  def forward(self, x):
    """ x: (batch, length, chan) """
    x = x.permute(0, 2, 1) # (batch, chan, length)
    y = self.conv(x)
    y = y.permute(0, 2, 1) # (batch, newlength, chan)
    return y

class VecConv1d(nn.Module):
  def __init__(self, chan, kernsz, edges_to_nodes=False):
    super().__init__()
    if edges_to_nodes:
      assert kernsz % 2 == 0
      self.conv = nn.Conv1d(chan, chan, kernsz, padding=(kernsz//2), bias=False)
    else:
      assert kernsz % 2 == 1
      self.conv = nn.Conv1d(chan, chan, kernsz, padding="same", bias=False)
  def forward(self, x):
    """ x: (batch, length, chan, 3) """
    batch, length, chan, must_be[3] = x.shape
    x = x.permute(0, 3, 2, 1).reshape(3*batch, chan, length) # (batch*3, chan, length)
    y = self.conv(x)
    must_be[batch*3], must_be[chan], newlength = y.shape # length might have changed!
    y = y.reshape(batch, 3, chan, newlength).permute(0, 3, 2, 1) # (batch, newlength, chan, 3)
    return y

class EdgeRelativeEmbedMLPPath(nn.Module):
  """ input embedding for edges, where 2 positions are passed as input.
      assumes graph structure is path """
  def __init__(self, adim, vdim):
    super().__init__()
    self.lin_v = VecLinear(4, vdim)
    self.scalar_layers = nn.Sequential(
      nn.Linear(4, adim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(adim, adim))
    self.conv_v = VecConv1d(vdim, 4, True)
    self.conv_a = ScalConv1d(adim, 4, True)
  def forward(self, pos_0, pos_1):
    """ pos0: (batch, nodes, 3)
        pos1: (batch, nodes, 3)
        return: tuple(a_out, v_out)
        a_out: (batch, nodes, adim)
        v_out: (batch, nodes, vdim, 3) """
    batch,          nodes,          must_be[3] = pos_0.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_1.shape
    vecs = torch.stack([ # 4 relative vectors
        pos_0[:,  1:] - pos_0[:, :-1],
        pos_1[:,  1:] - pos_1[:, :-1],
        pos_1[:, :-1] - pos_0[:,  1:],
        pos_1[:,  1:] - pos_0[:, :-1],
      ], dim=2) # (batch, nodes - 1, 4, 3)
    norms = torch.linalg.vector_norm(vecs, dim=-1) # (batch, nodes - 1, 4)
    y_a = self.scalar_layers(norms)
    y_v = self.lin_v(vecs)
    a_out = self.conv_a(y_a)
    v_out = self.conv_v(y_v)
    return a_out, v_out

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
    z_a = self.gnorm_a(self.conv_1_a(y_a))
    z_v = self.gnorm_v(self.conv_1_v(y_v))
    return pos_0, pos_1, (x_a + z_a), (x_v + z_v)

class Meanpred(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.lin_v_vel = VecLinear(vdim, 1)
    self.lin_v_pos = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def _predict(self, pos0, vel0):
    x_a, x_v = self.node_enc(pos0, pos0 + vel0)
    *_, y_v = self.blocks((pos0, pos0 + vel0, x_a, x_v))
    return self.lin_v_pos(y_v)[:, :, 0]*self.out_norm_coeff, self.lin_v_vel(y_v)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, vel0):
    pred_dpos, pred_vel = self._predict(pos0, vel0)
    return pos0 + pred_dpos, pred_vel


class MeanpredModel:
  def __init__(self, meanpred, config):
    self.meanpred = meanpred
    self.config = config
    assert config.sim.space_dim == 3
    assert config.cond_type == Condition.COORDS
    assert config.subtract_mean == 0
    assert not config.x_only
    self.n_nodes = config.sim.poly_len
    self.init_optim()
  def init_optim(self):
    self.optim = torch.optim.AdamW(self.meanpred.parameters(),
      self.config["lr"],
      (self.config["beta_1"], self.config["beta_2"]),
      weight_decay=self.config["weight_decay"])
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
  def _to_internal_shape(self, tens):
    """ convert tens to the shape that is used internally by this class """
    batch, must_be[2*self.n_nodes*3] = tens.shape
    tens = tens.reshape(batch, 2, self.n_nodes, 3)
    x, v = tens[:, 0], tens[:, 1]
    return x, v
  def _to_external_shape(self, x, v):
    """ convert tensors x, v to the shape that is used externally """
    batch,          must_be[self.n_nodes], must_be[3] = x.shape
    must_be[batch], must_be[self.n_nodes], must_be[3] = v.shape
    tens = torch.cat([x, v], dim=1)
    return tens.reshape(batch, 2*self.n_nodes*3)
  def train_step(self, data, cond):
    data, cond = self._to_internal_shape(data), self._to_internal_shape(cond)
    x0, v0 = cond
    x1, v1 = data
    return self._train_step(x0, v0,  x1, v1)
  def _train_step(self, x0, v0,  x1, v1):
    self.optim.zero_grad()
    x1_hat, v1_hat = self.meanpred(x0, v0)
    loss_x = ((x1_hat - x1)**2).sum(2).mean()*self.config["loss_x_scale"]
    loss_v = ((v1_hat - v1)**2).sum(2).mean()
    loss = loss_x + loss_v
    loss.backward()
    self.optim.step()
    return loss_x, loss_v
  def predict(self, cond):
    x0, v0 = self._to_internal_shape(cond)
    with torch.no_grad():
      x_hat, v_hat = self.meanpred(x0, v0)
      return self._to_external_shape(x_hat, v_hat)
  def set_eval(self, cond):
    if cond:
      self.meanpred.eval()
    else:
      self.meanpred.train()



class MeanpredTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    N, L, state_dim = trajs.shape
    cond = self.model.config.cond(trajs[:, :-1].reshape(N*(L - 1), state_dim))
    data = trajs[:, 1:].reshape(N*(L - 1), state_dim)
    loss_x, loss_v = self.model.train_step(data, cond)
    loss = loss_x + loss_v
    print(f"{i}\t ℒx = {loss_x:05.6f}\t ℒv = {loss_v:05.6f}\t ℒ = {loss:05.6f}")
    self.board.scalar("loss_x", i, loss_x)
    self.board.scalar("loss_v", i, loss_v)
    self.board.scalar("loss", i, loss)



# export model class and trainer class:
modelclass   = MeanpredModel
trainerclass = MeanpredTrainer








