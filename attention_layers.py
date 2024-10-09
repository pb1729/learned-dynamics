import torch
import torch.nn as nn

from utils import must_be
from triton_kernels.proxattn import proxattn
from layers_common import VecRootS


class ProbePoints(nn.Module):
  """ Create probe points for use in a ProximityAttention layer.
      In order to make probe points translation-invariant, we take them to be:
      - A normalized mixture of a number of base-points in space.
      - Plus any translation-invariant vectors we want. """
  def __init__(self, n_base_pts, n_probe_pts, adim, vdim):
    super().__init__()
    self.n_base_pts = n_base_pts
    self.n_probe_pts = n_probe_pts
    self.lin_mix = nn.Linear(adim, n_base_pts*n_probe_pts, bias=False)
    self.W = nn.Parameter(torch.empty(n_probe_pts, vdim))
  def self_init(self):
    must_be[self.n_probe_pts], vdim = self.W.shape
    nn.init.normal_(self.W, std=(vdim**(-0.5)))
  def forward(self, ax, vx, positions):
    """ ax: (batch, nodes, adim)
        vx: (batch, nodes, vdim, 3)
        positions: list of (batch, nodes, 3)
        ans: (n_probe_pts, batch, nodes, 3) """
    batch,          nodes,          adim = ax.shape
    must_be[batch], must_be[nodes], vdim, must_be[3] = vx.shape
    positions = torch.stack(positions, dim=2)
    must_be[batch], must_be[nodes], must_be[self.n_base_pts], must_be[3] = positions.shape
    mixture_factors = self.lin_mix(ax).reshape(batch, nodes, self.n_base_pts, self.n_probe_pts)
    mixture_factors = torch.softmax(mixture_factors, dim=2)
    probe_pts = torch.einsum("bnsv, bnsp -> pbnv", positions, mixture_factors) # (n_probe_pts, batch, nodes, 3)
    delta_v = torch.einsum("pj, bnjv -> pbnv", self.W, vx) # (n_probe_pts, batch, nodes, 3)
    probe_pts = probe_pts + delta_v
    return probe_pts


class ProbePoints2(nn.Module):
  """ Create probe points for use in a ProximityAttention layer.
      In order to make probe points translation-invariant, we take them to be:
      - A normalized mixture of a number of base-points in space.
      - Plus any translation-invariant vectors we want. """
  def __init__(self, n_base_pts, heads, adim, vdim):
    super().__init__()
    self.n_base_pts = n_base_pts
    self.H = heads
    self.lin_mix = nn.Linear(adim, n_base_pts*heads, bias=False)
    self.W_mix = nn.Parameter(torch.empty(heads, n_base_pts, adim))
    self.W_vec = nn.Parameter(torch.empty(heads, vdim))
  def self_init(self):
    must_be[self.H], vdim = self.W_vec.shape
    nn.init.normal_(self.W_vec, std=(vdim**(-0.5)))
    must_be[self.H], must_be[self.n_base_pts], adim = self.W_mix.shape
    nn.init.normal_(self.W_mix, std=(adim**(-0.5)))
  def forward(self, ax, vx, positions):
    """ ax: (batch, nodes, adim)
        vx: (batch, nodes, vdim, 3)
        positions: list of (batch, nodes, 3)
        ans: (batch, heads, nodes, 3) """
    batch,          nodes,          adim = ax.shape
    must_be[batch], must_be[nodes], vdim, must_be[3] = vx.shape
    positions = torch.stack(positions, dim=2) # (batch, nodes, n_base_pts, 3)
    must_be[batch], must_be[nodes], must_be[self.n_base_pts], must_be[3] = positions.shape
    mixture_factors = torch.einsum("hpj, bnj -> bhnp", self.W_mix, ax)
    mixture_factors = torch.softmax(mixture_factors, dim=3)
    probe_pts = torch.einsum("bnpv, bhnp -> bhnv", positions, mixture_factors) # (batch, heads, nodes, 3)
    delta_v = torch.einsum("hj, bnjv -> bhnv", self.W_vec, vx) # (batch, heads, nodes, 3)
    probe_pts = probe_pts + delta_v
    return probe_pts


class ProbePoints3(nn.Module):
  """ Even simpler probe points layer! """
  def __init__(self, heads, vdim):
    super().__init__()
    self.H = heads
    self.W_vec = nn.Parameter(torch.empty(heads, vdim))
  def self_init(self):
    must_be[self.H], vdim = self.W_vec.shape
    nn.init.normal_(self.W_vec, std=(vdim**(-0.5)))
  def forward(self, vx, pos):
    """ vx: (batch, nodes, vdim, 3)
        pos: (batch, nodes, 3)
        ans: (batch, heads, nodes, 3) """
    batch, nodes, vdim, must_be[3] = vx.shape
    must_be[batch], must_be[nodes], must_be[3] = pos.shape
    delta_v = torch.einsum("hj, bnjv -> bhnv", self.W_vec, vx) # (batch, heads, nodes, 3)
    return pos[:, None] + delta_v


class ProximityAttention(nn.Module):
  """ A multiheaded attention module. The layer can be configured with the following parameters:
      r0_list: (H)        --> a python list of characteristic radii for each attention head
      kq_dim: (dim, dim)  --> tuple of adim and vdim for keys and queries
      chan: (dim, dim)    --> tuple of adim and vdim for input and output data
      kq_prescale: ()     --> k*q dot products are multiplied by this to make them smaller
      Attention is based both on vector dot products, and also on spatial proximity.
      This implementation does not allow masking. """
  def __init__(self, r0_list, kq_dim, chan, kq_prescale=0.01, kq_scale=0.1):
    super().__init__()
    self.H = len(r0_list)
    self.register_buffer("r0", torch.tensor(r0_list), persistent=False)
    kq_adim, kq_vdim = kq_dim
    achan, vchan = chan
    self.kq_ascale = (kq_prescale/kq_adim)**0.5
    self.kq_vscale = (kq_prescale/kq_vdim)**0.5
    self.kq_scale = kq_scale
    assert achan % self.H == 0, "channels must be evenly divisible by number of heads"
    assert vchan % self.H == 0, "channels must be evenly divisible by number of heads"
    self.W_ak = nn.Parameter(torch.empty(self.H, kq_adim, achan))
    self.W_vk = nn.Parameter(torch.empty(self.H, kq_vdim, vchan))
    self.W_aq = nn.Parameter(torch.empty(self.H, kq_adim, achan))
    self.W_vq = nn.Parameter(torch.empty(self.H, kq_vdim, vchan))
    avaldim, vvaldim = achan // self.H, vchan // self.H # divide by number of heads to get useful sizes for value weight matrices
    self.W_aval = nn.Parameter(torch.empty(self.H, avaldim, achan))
    self.W_vval = nn.Parameter(torch.empty(self.H, vvaldim, vchan))
    self.W_a2vval = nn.Parameter(torch.empty(self.H, vvaldim, achan)) # special weight matrix for getting displacement info
    self.W_v2aval = nn.Parameter(torch.empty(self.H, avaldim, vchan)) # special weight matrix for getting displacement info
  def _param_init_with_scale(self, param, scale=1.):
    must_be[self.H], fan_in, fan_out = param.shape
    std = scale*(fan_in**(-0.5))
    nn.init.normal_(param, std=std)
  def self_init(self):
    self._param_init_with_scale(self.W_ak, self.kq_ascale)
    self._param_init_with_scale(self.W_vk, self.kq_vscale)
    self._param_init_with_scale(self.W_aq, self.kq_ascale)
    self._param_init_with_scale(self.W_vq, self.kq_vscale)
    self._param_init_with_scale(self.W_aval)
    self._param_init_with_scale(self.W_vval)
    self._param_init_with_scale(self.W_a2vval)
    self._param_init_with_scale(self.W_v2aval)
  def forward(self, x, pos_k, pos_q):
    """ performs a proximity attention operation.
        x is the input data vector
        pos_k, pos_q are the spatial locations of the key and query probe points
        x: tuple(ax, vx)
          ax: (batch, nodes, achan)
          vx: (batch, nodes, vchan, 3)
        pos_k, pos_q: (batch, nodes, 3)
        Attention score is given by softmax(k*q)*(r0**2 / (r0**2 + r**2))
        Data about separations between probe points is mixed into the output.
        ans: tuple(aans, vans)
          aans: (batch, nodes, achan)
          vans: (batch, nodes, vchan, 3) """
    ax, vx = x
    batch,          nodes,          achan = ax.shape
    must_be[batch], must_be[nodes], vchan, must_be[3] = vx.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_k.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_q.shape
    # compute basic attention:
    ak = torch.einsum("hij, bnj -> bnhi",   self.W_ak, ax) # (batch, nodes, H, kq_adim)
    vk = torch.einsum("hij, bnjv -> bnhiv", self.W_vk, vx) # (batch, nodes, H, kq_vdim, 3)
    aq = torch.einsum("hij, bnj -> bnhi",   self.W_aq, ax) # (batch, nodes, H, kq_adim)
    vq = torch.einsum("hij, bnjv -> bnhiv", self.W_vq, vx) # (batch, nodes, H, kq_vdim, 3)
    akq_dot = torch.einsum("bnhi, bmhi -> bnmh",   ak, aq) # (batch, nodes, nodes, H)
    vkq_dot = torch.einsum("bnhiv, bmhiv -> bnmh", vk, vq) # (batch, nodes, nodes, H)
    dot = self.kq_scale*(akq_dot + vkq_dot) # (batch, nodes, nodes, H)
    # compute proximity:
    separation = pos_k[:, None, :] - pos_q[:, :, None] # (batch, nodes, nodes, 3)
    dist_sq = (separation**2).sum(3) # (batch, nodes, nodes) squared-distance matrix
    r0_sq = self.r0**2
    proximity = r0_sq/(r0_sq + dist_sq[..., None]) # (batch, nodes, nodes, H)
    directions = separation[:, :, :, None]*torch.sqrt(proximity/r0_sq)[:, :, :, :, None] # (batch, nodes, nodes, H, 3) max magnitude is 1
    dot = dot + torch.log(proximity) # update attention pre-activations with proximity
    # attention operation
    attention = torch.softmax(dot, dim=1) # dim 1 corresponds to "which key?"
    aans = torch.einsum("bnmh, hij, bnj -> bmhi",   attention, self.W_aval, ax) # (batch, nodes, H, avaldim)
    vans = torch.einsum("bnmh, hij, bnjv -> bmhiv", attention, self.W_vval, vx) # (batch, nodes, H, vvaldim, 3)
    # special value messages using spatial displacements:
    aans = aans + torch.einsum("bnmh, hij, bnjv, bnmhv -> bmhi", attention, self.W_v2aval, vx, directions)
    vans = vans + torch.einsum("bnmh, hij, bnj, bnmhv -> bmhiv", attention, self.W_a2vval, ax, directions)
    return aans.reshape(batch, nodes, achan), vans.reshape(batch, nodes, vchan, 3)



class ProximityAttentionV2(nn.Module):
  """ A multiheaded attention module. The layer can be configured with the following parameters:
      r0_list: (H)        --> a python list of characteristic radii for each attention head
      kq_dim: (dim, dim)  --> tuple of adim and vdim for keys and queries
      chan: (dim, dim)    --> tuple of adim and vdim for input and output data
      kq_prescale: ()     --> k*q dot products are multiplied by this to make them smaller
      Attention is based both on vector dot products, and also on spatial proximity.
      This implementation does not allow masking.

      This is version 2, with radial separation encodings added onto the attention operation. """
  def __init__(self, r0_list, kq_dim, chan, kq_prescale=0.01, kq_scale=0.1):
    super().__init__()
    self.H = len(r0_list)
    self.register_buffer("r0", torch.tensor(r0_list), persistent=False)
    kq_adim, kq_vdim = kq_dim
    achan, vchan = chan
    self.kq_ascale = (kq_prescale/kq_adim)**0.5
    self.kq_vscale = (kq_prescale/kq_vdim)**0.5
    self.kq_scale = kq_scale
    assert achan % self.H == 0, "channels must be evenly divisible by number of heads"
    assert vchan % self.H == 0, "channels must be evenly divisible by number of heads"
    self.W_ak = nn.Parameter(torch.empty(self.H, kq_adim, achan))
    self.W_vk = nn.Parameter(torch.empty(self.H, kq_vdim, vchan))
    self.W_aq = nn.Parameter(torch.empty(self.H, kq_adim, achan))
    self.W_vq = nn.Parameter(torch.empty(self.H, kq_vdim, vchan))
    avaldim, vvaldim = achan // self.H, vchan // self.H # divide by number of heads to get useful sizes for value weight matrices
    self.W_aval = nn.Parameter(torch.empty(self.H, avaldim, achan))
    self.W_vval = nn.Parameter(torch.empty(self.H, vvaldim, vchan))
    self.W_a2vval = nn.Parameter(torch.empty(self.H, vvaldim, achan)) # special weight matrix for getting displacement info
    self.W_v2aval = nn.Parameter(torch.empty(self.H, avaldim, vchan)) # special weight matrix for getting displacement info
    self.W_aradenc = nn.Parameter(torch.empty(self.H, avaldim, 16)) # special weight matrix for radial encodings
    self.W_vradenc = nn.Parameter(torch.empty(self.H, vvaldim, 16)) # special weight matrix for radial encodings
  def _param_init_with_scale(self, param, scale=1.):
    must_be[self.H], fan_in, fan_out = param.shape
    std = scale*(fan_in**(-0.5))
    nn.init.normal_(param, std=std)
  def self_init(self):
    self._param_init_with_scale(self.W_ak, self.kq_ascale)
    self._param_init_with_scale(self.W_vk, self.kq_vscale)
    self._param_init_with_scale(self.W_aq, self.kq_ascale)
    self._param_init_with_scale(self.W_vq, self.kq_vscale)
    self._param_init_with_scale(self.W_aval)
    self._param_init_with_scale(self.W_vval)
    self._param_init_with_scale(self.W_a2vval)
    self._param_init_with_scale(self.W_v2aval)
    self._param_init_with_scale(self.W_aradenc)
    self._param_init_with_scale(self.W_vradenc)
  def forward(self, x, pos_k, pos_q):
    """ performs a proximity attention operation.
        x is the input data vector
        pos_k, pos_q are the spatial locations of the key and query probe points
        x: tuple(ax, vx)
          ax: (batch, nodes, achan)
          vx: (batch, nodes, vchan, 3)
        pos_k, pos_q: (batch, nodes, 3)
        Attention score is given by softmax(k*q)*(r0**2 / (r0**2 + r**2))
        Data about separations between probe points is mixed into the output.
        ans: tuple(aans, vans)
          aans: (batch, nodes, achan)
          vans: (batch, nodes, vchan, 3) """
    ax, vx = x
    batch,          nodes,          achan = ax.shape
    must_be[batch], must_be[nodes], vchan, must_be[3] = vx.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_k.shape
    must_be[batch], must_be[nodes], must_be[3] = pos_q.shape
    # compute basic attention:
    ak = torch.einsum("hij, bnj -> bnhi",   self.W_ak, ax) # (batch, nodes, H, kq_adim)
    vk = torch.einsum("hij, bnjv -> bnhiv", self.W_vk, vx) # (batch, nodes, H, kq_vdim, 3)
    aq = torch.einsum("hij, bnj -> bnhi",   self.W_aq, ax) # (batch, nodes, H, kq_adim)
    vq = torch.einsum("hij, bnjv -> bnhiv", self.W_vq, vx) # (batch, nodes, H, kq_vdim, 3)
    akq_dot = torch.einsum("bnhi, bmhi -> bnmh",   ak, aq) # (batch, nodes, nodes, H)
    vkq_dot = torch.einsum("bnhiv, bmhiv -> bnmh", vk, vq) # (batch, nodes, nodes, H)
    dot = self.kq_scale*(akq_dot + vkq_dot) # (batch, nodes, nodes, H)
    # compute proximity:
    separation = pos_k[:, None, :] - pos_q[:, :, None] # (batch, nodes, nodes, 3)
    dist_sq = (separation**2).sum(3) # (batch, nodes, nodes) squared-distance matrix
    r0_sq = self.r0**2
    proximity = r0_sq/(r0_sq + dist_sq[..., None]) # (batch, nodes, nodes, H)
    directions = separation[:, :, :, None]*torch.sqrt(proximity/r0_sq)[:, :, :, :, None] # (batch, nodes, nodes, H, 3) max magnitude is 1
    dot = dot + torch.log(proximity) # update attention pre-activations with proximity
    # attention operation
    attention = torch.softmax(dot, dim=1) # dim 1 corresponds to "which key?"
    aans = torch.einsum("bnmh, hij, bnj -> bmhi",   attention, self.W_aval, ax) # (batch, nodes, H, avaldim)
    vans = torch.einsum("bnmh, hij, bnjv -> bmhiv", attention, self.W_vval, vx) # (batch, nodes, H, vvaldim, 3)
    # special value messages using spatial displacements:
    aans = aans + torch.einsum("bnmh, hij, bnjv, bnmhv -> bmhi", attention, self.W_v2aval, vx, directions)
    vans = vans + torch.einsum("bnmh, hij, bnj, bnmhv -> bmhiv", attention, self.W_a2vval, ax, directions)
    # version 2 addition: radial encodings!
    relative_dist = torch.sqrt(dist_sq[..., None]/r0_sq)[..., None] # (batch, nodes, nodes, H, 1)
    ω = torch.arange(8, device=relative_dist.device)*0.5
    radenc = torch.cat([torch.sin(ω*relative_dist), torch.cos(ω*relative_dist)], dim=-1) # (batch, nodes, nodes, H, 16)
    aans = aans + torch.einsum("bnmh, hir, bnmhr -> bmhi", attention, self.W_aradenc, radenc) # (batch, nodes, H, avaldim)
    vans = vans + torch.einsum("bnmh, hir, bnmhr, bnmhv -> bmhiv", attention, self.W_vradenc, radenc, directions)
    return aans.reshape(batch, nodes, achan), vans.reshape(batch, nodes, vchan, 3)


class ProximityFlashAttention(nn.Module):
  def __init__(self, r0_list, adim, vdim, chan, vec_actv_class=VecRootS):
    super().__init__()
    self.H = len(r0_list)
    self.chan = chan
    self.register_buffer("r0sq", torch.tensor(r0_list)**2, persistent=False)
    self.W_aq = nn.Parameter(torch.empty(self.H, chan, adim))
    self.W_vq = nn.Parameter(torch.empty(self.H, chan, vdim))
    self.W_ak = nn.Parameter(torch.empty(self.H, chan, adim))
    self.W_vk = nn.Parameter(torch.empty(self.H, chan, vdim))
    self.W_av = nn.Parameter(torch.empty(self.H, chan, adim))
    self.W_vv = nn.Parameter(torch.empty(self.H, chan, vdim))
    self.W_ao = nn.Parameter(torch.empty(self.H, adim, chan))
    self.W_vo = nn.Parameter(torch.empty(self.H, vdim, chan))
    self.vec_actv = vec_actv_class()
  def _param_init_with_scale(self, param, scale=1.):
    must_be[self.H], fan_in, fan_out = param.shape
    std = scale*(fan_in**(-0.5))
    nn.init.normal_(param, std=std)
  def self_init(self):
    self._param_init_with_scale(self.W_aq)
    self._param_init_with_scale(self.W_vq)
    self._param_init_with_scale(self.W_ak)
    self._param_init_with_scale(self.W_vk)
    self._param_init_with_scale(self.W_av)
    self._param_init_with_scale(self.W_vv)
  def stack_av(self, a, v):
    """ stack scalars and vectors
        a: (..., chan)
        v: (..., chan, 3)
        out: (..., 4*chan) """
    *rest, chan = a.shape
    *must_be[rest], must_be[chan], must_be[3] = v.shape
    return torch.cat([a[..., None], v], dim=-1).reshape(*rest, 4*chan)
  def split_av(self, tens):
    """ reverse of stack_av """
    *rest, chan4 = tens.shape
    chan = chan4 // 4
    return tens[..., :chan], tens[..., chan:].reshape(*rest, chan, 3)
  def forward(self, ax, vx, pos_k, pos_q):
    """ simple version of proximity attention
        ax, ay: (batch, nodes, adim)
        vx, vy: (batch, nodes, vdim, 3)
        pos_k, pos_q: (batch, heads, nodes, 3) """
    batch, nodes, adim = ax.shape
    must_be[batch], must_be[nodes], vdim, must_be[3] = vx.shape
    must_be[batch], must_be[self.H], must_be[nodes], must_be[3] = pos_k.shape
    must_be[batch], must_be[self.H], must_be[nodes], must_be[3] = pos_q.shape
    # get Q, K, V matrices
    Q = self.stack_av(torch.einsum("hij, bnj -> bhni", self.W_aq, ax), torch.einsum("hij, bnjv -> bhniv", self.W_vq, vx)).reshape(batch*self.H, nodes, 4*self.chan)
    K = self.stack_av(torch.einsum("hij, bnj -> bhni", self.W_ak, ax), torch.einsum("hij, bnjv -> bhniv", self.W_vk, vx)).reshape(batch*self.H, nodes, 4*self.chan)
    V = self.stack_av(torch.einsum("hij, bnj -> bhni", self.W_av, ax), torch.einsum("hij, bnjv -> bhniv", self.W_vv, vx)).reshape(batch*self.H, nodes, 4*self.chan)
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    pos_q, pos_k = pos_q.contiguous(), pos_k.contiguous()
    # prevent q,k values from getting too large
    Q = self.vec_actv(Q) # TODO: this is along a dimension of size 4*self.chan, not 3!
    K = self.vec_actv(K) # consider if this is maybe really dumb (idk, could be fine...)
    # do attention
    O = proxattn(Q, K, V, pos_q.reshape(batch*self.H, nodes, 3), pos_k.reshape(batch*self.H, nodes, 3), self.r0sq)
    O /= nodes**0.5 # prevent runaway activations...
    ay, vy = self.split_av(O)
    ay = torch.einsum("hji, bhni -> bnj", self.W_ao, ay.reshape(batch, self.H, nodes, self.chan))
    vy = torch.einsum("hji, bhniv -> bnjv", self.W_vo, vy.reshape(batch, self.H, nodes, self.chan, 3))
    return ay, vy




if __name__ == "__main__":
  device = "cuda"
  batch = 32
  nodes = 24
  adim = 88
  vdim = 44
  pp = ProbePoints(2, 2, adim, vdim)
  pp.self_init()
  pp.to(device)
  # m = ProximityAttention([0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16.], (16, 12), (88, 44))
  m = ProximityAttentionV2([0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16.], (16, 12), (88, 44))
  m.self_init()
  m.to(device)
  ax = torch.randn(batch, nodes, adim, device=device)
  vx = torch.randn(batch, nodes, vdim, 3, device=device)*0.577
  pos_0 = torch.randn(batch, nodes, 3, device=device)*3.2
  pos_1 = torch.randn(batch, nodes, 3, device=device)*3.2
  print("calculating probe points...")
  pos_kq = pp(ax, vx, [pos_0, pos_1])
  translate = 10*torch.randn(3, device=device)
  pos_0_prime = pos_0 + translate
  pos_1_prime = pos_1 + translate
  pos_kq_prime = pp(ax, vx, [pos_0_prime, pos_1_prime])
  print(((pos_kq + translate - pos_kq_prime)**2).max())
  pos_k = pos_kq[0]
  pos_q = pos_kq[1]
  print("forward pass...")
  aans, vans = m((ax, vx), pos_k, pos_q)
  aans_prime, vans_prime = m((ax, vx), pos_kq_prime[0], pos_kq_prime[1])
  print((aans**2).mean(), (vans**2).mean())
  print(((aans - aans_prime)**2).max(), ((vans - vans_prime)**2).sum(-1).max())
