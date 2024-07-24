import torch
from torch.optim import Optimizer


class FriendlyAverage34(Optimizer):
  """ Implements the friendly average optimizer, with 1/4, 1/4, 1/4 averaging.
      You can set learning rate, beta (which determines the averaging for second moment)
      an eps value for numerical stability, and weight decay. Weight decay mixes the weights
      with noise as they decay. (Setting weight-decay to 0 also means there will be no noise.) """
  def __init__(self, params, lr=0.01, beta=0.999, eps=1e-6, weight_decay=None):
    assert eps >= 0., "invalid epsilon value"
    assert 0. <= beta <= 1., "beta should be between 0 and 1"
    defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
    super().__init__(params, defaults)
  def _get_state(self, p, weight_decay):
    """ get state, initialize if necessary """
    ans = self.state[p]
    if len(ans) == 0:
      ans["step"] = 0
      ans["p0"] = torch.clone(p.detach())
      ans["k1"] = torch.zeros_like(p)
      ans["k2"] = torch.zeros_like(p)
      ans["v"]  = torch.zeros_like(p)
      if weight_decay is not None and p.numel() > 1:
        ans["base_var"] = torch.cov(p.flatten())
      else:
        ans["base_var"] = None
    return ans
  def _step(self):
    for group in self.param_groups:
      lr, beta, eps, weight_decay = group["lr"], group["beta"], group["eps"], group["weight_decay"]
      for param in group["params"]:
        state = self._get_state(param, weight_decay)
        state["step"] += 1
        grad = param.grad if param.grad is not None else torch.tensor(0., device=param.device)
        # estimate second moment:
        state["v"].mul_(beta)
        state["v"].addcmul_(grad, grad, value=(1. - beta))
        v_hat = state["v"]/(1. - beta**state["step"]) # bias correction
        # get the current k value:
        k = -lr*grad/(torch.sqrt(v_hat) + eps)
        # update parameter:
        phase = state["step"] % 3
        if phase == 0:
          param.copy_(state["p0"])
          param.add_(0.25*(state["k1"] + state["k2"] + k))
          if state["base_var"] is not None:
            param.mul_(1. - lr*weight_decay)
            param.add_(torch.randn_like(param)*torch.sqrt(2*lr*weight_decay*state["base_var"]))
          state["p0"].copy_(param)
        elif phase == 1:
          param.copy_(state["p0"])
          param.add_(k)
          state["k1"].copy_(k)
        else: # phase == 2
          param.copy_(state["p0"])
          param.add_(k)
          state["k2"].copy_(k)
  def step(self, closure=None):
    with torch.no_grad():
      self._step()






