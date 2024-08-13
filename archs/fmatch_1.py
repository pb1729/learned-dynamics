import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batched_model_eval
from config import Config, Condition
from layers_common import *
from attention_layers import *
from vamp_score import vamp_score, vamp_score3d


#
# Feature matching GAN based on the VAMP score
# Interesting stuff here is the special-purpose code for computing the VAMP score
# on vector features (not only scalar features like before). In principle, we could
# also go to 5-dimensional l=2 irreps, or even higher values of l.
#


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
    self.probe_pts = ProbePoints(2, 2, adim, vdim)
    self.prox_attn = ProximityAttention(config["r0_list"], config["kq_dim"], (adim, vdim))
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
    probes = self.probe_pts(y_a, y_v, [pos_0, pos_1])
    probes_k, probes_q = probes[0], probes[1]
    dy_a, dy_v = self.prox_attn((y_a, y_v), probes_k, probes_q)
    y_a, y_v = y_a + dy_a, y_v + dy_v
    z_a = self.gnorm_a(self.conv_1_a(y_a))
    z_v = self.gnorm_v(self.conv_1_v(y_v))
    return pos_0, pos_1, (x_a + z_a), (x_v + z_v)


class VAMPNet3D(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.x_only, "still need to implement having absolute velocity, so can't do x_only=False yet"
    assert config.sim.space_dim == 3
    adim, vdim = config["adim"], config["vdim"]
    anf, vnf = config["anf"], config["vnf"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config),
      Block(config))
    self.lin_a = nn.Linear(config.sim.poly_len*adim, anf)
    self.lin_v = VecLinear(config.sim.poly_len*vdim, vnf)
  def forward(self, pos):
    """ decoder forward pass. returns estimated mean value of state """
    batch, poly_len, must_be[3] = pos.shape
    x_a, x_v = self.node_enc(pos, pos) # encode with same position twice, a little wasteful! (TODO)
    *_, y_a, y_v = self.blocks((pos, pos, x_a, x_v)) # ditto
    return self.lin_a(y_a.reshape(batch, -1)), self.lin_v(y_v.reshape(batch, -1, 3))


class Generator(nn.Module):
  def __init__(self, config):
    super().__init__()
    adim, vdim = config["adim"], config["vdim"]
    self.node_enc = NodeRelativeEmbedMLP(adim, vdim)
    self.blocks = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    self.blocks_tune = nn.Sequential(
      Block(config),
      Block(config),
      Block(config))
    # some wastage of compute here, since the final sets of scalar values are not really used
    self.lin_v_node = VecLinear(vdim, 1)
    self.lin_v_node_tune = VecLinear(vdim, 1)
    self.out_norm_coeff = vdim**(-0.5)
  def _predict(self, pos0, noised):
    x_a, x_v = self.node_enc(pos0, noised)
    *_, y_v = self.blocks((pos0, noised, x_a, x_v))
    return self.lin_v_node(y_v)[:, :, 0]*self.out_norm_coeff
  def _finetune(self, pos0, noised, ε_a, ε_v):
    x_a, x_v = self.node_enc(pos0, noised)
    x_a = x_a + ε_a
    x_v = x_v + ε_v
    *_, y_v = self.blocks_tune((pos0, noised, x_a, x_v))
    return self.lin_v_node_tune(y_v)[:, :, 0]*self.out_norm_coeff
  def forward(self, pos0, noise, ε_a, ε_v):
    noised = pos0 + noise
    pred_noise = self._predict(pos0, noised)
    noised = noised - pred_noise
    return noised + self._finetune(pos0, noised, ε_a, ε_v)


class KoopmanFMatch:
  """ class containing a VAMPNet and a generator trained by feature matching """
  is_gan = True
  def __init__(self, model, gen, config):
    self.model = model
    self.gen = gen
    self.config = config
    self.init_optim()
  def init_optim(self):
    self.optim = torch.optim.Adam(self.model.parameters(),
      self.config["lr"], (self.config["beta_1"], self.config["beta_2"]))
    self.optim_g = torch.optim.Adam(self.gen.parameters(),
      self.config["lr"], (self.config["beta_1"], self.config["beta_2"]))
  @staticmethod
  def load_from_dict(states, config):
    model = VAMPNet3D(config).to(config.device)
    gen = Generator(config).to(config.device)
    model.load_state_dict(states["vampnet"])
    gen.load_state_dict(states["gen"])
    return KoopmanFMatch(model, gen, config)
  @staticmethod
  def makenew(config):
    model = VAMPNet3D(config).to(config.device)
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
    loss_a, loss_v, loss = self.train_step_vamp(trajs)
    loss_g = self.train_step_gen(trajs)
    return loss_a, loss_v, loss, loss_g
  def train_step_vamp(self, trajs):
    batch, L, must_be[self.config.sim.poly_len*3] = trajs.shape
    chi_a, chi_v = self.model(trajs.reshape(batch*L, self.config.sim.poly_len, 3))
    chi_a = chi_a.reshape(batch, L, self.config["anf"])
    chi_v = chi_v.reshape(batch, L, self.config["vnf"], 3)
    _, trans_1_a, score_a = vamp_score(
      chi_a[:, :-1].reshape(batch*(L - 1), -1), chi_a[:, 1:].reshape(batch*(L - 1), -1),
      mode="trans")
    _, trans_1_v, score_v = vamp_score3d(
      chi_v[:, :-1].reshape(batch*(L - 1), -1, 3), chi_v[:, 1:].reshape(batch*(L - 1), -1, 3),
      mode="trans")
    self.trans_1_a, self.trans_1_v = trans_1_a.detach(), trans_1_v.detach() # detach and store these for later use
    loss = -(score_a + score_v)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    return -score_a.item(), -score_v.item(), loss.item()
  def train_step_gen(self, trajs):
    # get true features
    batch, L, must_be[self.config.sim.poly_len*3] = trajs.shape
    chi_a, chi_v = self.model(trajs.reshape(batch*L, self.config.sim.poly_len, 3))
    chi_a = chi_a.reshape(batch, L, self.config["anf"])
    chi_v = chi_v.reshape(batch, L, self.config["vnf"], 3)
    chi_a = chi_a[:, 1:].reshape(batch*(L - 1), -1)
    chi_v = chi_v[:, 1:].reshape(batch*(L - 1), -1, 3)
    # get predicted features
    x_pred = self.generate(trajs[:, :-1].reshape(batch*(L - 1), self.config.sim.poly_len, 3))
    chi_pred_a, chi_pred_v = self.model(x_pred)
    # convert to the nice basis
    chi_a = self.trans_1_a(chi_a)
    chi_v = self.trans_1_v(chi_v)
    chi_pred_a = self.trans_1_a(chi_pred_a)
    chi_pred_v = self.trans_1_v(chi_pred_v)
    # feature matching loss (MSE):
    loss = ((chi_a - chi_pred_a)**2).mean() + ((chi_v - chi_pred_v)**2).sum(2).mean()
    self.optim_g.zero_grad()
    loss.backward()
    self.optim_g.step()
    return loss.item()
  def generate(self, cond):
    batch, must_be[self.config.sim.poly_len], must_be[3] = cond.shape
    pos_noise, z_a, z_v = self.get_latents(batch)
    return self.gen(cond, pos_noise, z_a, z_v)
  def get_latents(self, batchsz):
    """ sample latent space for generator """
    pos_noise = self.config["z_scale"]*torch.randn(batchsz, self.config.sim.poly_len, 3, device=self.config.device)
    z_a = torch.randn(batchsz, self.config.sim.poly_len, self.config["adim"], device=self.config.device)
    z_v = torch.randn(batchsz, self.config.sim.poly_len, self.config["vdim"], 3, device=self.config.device)
    return pos_noise, z_a, z_v
  def set_eval(self, bool_eval):
    if bool_eval:
      self.model.eval()
      self.gen.eval()
    else:
      self.model.train()
      self.gen.train()
  def predict(self, cond):
    batch, must_be[3*self.config.sim.poly_len] = cond.shape
    with torch.no_grad():
      return self.generate(cond.reshape(batch, self.config.sim.poly_len, 3)).reshape(batch, 3*self.config.sim.poly_len)


class KoopmanTrainer:
  def __init__(self, model, board):
    self.model = model
    self.board = board
  def step(self, i, trajs):
    loss_a, loss_v, loss, loss_g = self.model.train_step(trajs)
    print(f"{i}\t ℒ = {loss:05.6f} \t ℒᴬ = {loss_a:05.6f} \t ℒⱽ = {loss_v:05.6f} \t ℒᴳ = {loss_g:05.6f}")
    self.board.scalar("loss", i, loss)
    self.board.scalar("loss_a", i, loss_a)
    self.board.scalar("loss_v", i, loss_v)
    self.board.scalar("loss_g", i, loss_g)



# export model class and trainer class:
modelclass   = KoopmanFMatch
trainerclass = KoopmanTrainer




