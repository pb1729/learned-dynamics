import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import grad_record, must_be
from atoms_display import launch_atom_display


def magdata(x):
  x = x.detach()
  moment_1 = x.mean()
  moment_2 = (x**2).mean()
  intensity = moment_1**2/moment_2
  return torch.sqrt(moment_2).item(), intensity.item()

class TGTracker(nn.Module):
  """ Tensor/Gradient Tracker.
      Insert this class into various parts of your model... """
  all_trackers = [] # static list of all trackers that have been created
  def __init__(self, name, print_on_callback=False):
    super().__init__()
    self.x_mags,  self.x_intns  = [], []
    self.dx_mags, self.dx_intns = [], []
    self.all_trackers.append(self)
    self.name = name
    self.print_on_callback = print_on_callback
  def fwd_callback(self, x):
    mag, intn = magdata(x)
    self.x_mags.append(mag)
    self.x_intns.append(intn)
    if self.print_on_callback:
      print("FWD", self.name, mag, intn)
  def bwd_callback(self, dx):
    mag, intn = magdata(dx)
    self.dx_mags.append(mag)
    self.dx_intns.append(intn)
    if self.print_on_callback:
      print("BWD", self.name, mag, intn)
  def create_figure(self):
    fig = plt.figure()
    ax_x = fig.add_subplot(111)
    ax_dx = ax_x.twinx()
    # plots
    ax_x.scatter(range(len(self.x_mags)), self.x_mags, c=self.x_intns, cmap="winter", vmin=0, vmax=1)
    ax_dx.scatter(range(len(self.dx_mags)), self.dx_mags, c=self.dx_intns, cmap="autumn", vmin=0, vmax=1)
    # axis limits
    ax_x.set_ylim(0., None)
    ax_dx.set_ylim(0., None)
    # words
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("magnitude x")
    ax_dx.set_ylabel("magnitude grad x")
    ax_x.set_title(self.name)
  def forward(self, x):
    self.fwd_callback(x)
    return grad_record(x, lambda dx: self.bwd_callback(dx))
  @staticmethod
  def showall():
    for tracker in TGTracker.all_trackers:
      tracker.create_figure()
    plt.show()

class PosTracker(nn.Module):
  """ Class to track particle positions and their gradient. """
  def __init__(self, grad_scale=1.):
    super().__init__()
    self.grad_scale = grad_scale
    PosTracker.tracker = self
    self.display = None
  def forward(self, pos_0, pos_1):
    pos_0_np = pos_0[0].detach().cpu().numpy()
    pos_1_np = pos_1[0].detach().cpu().numpy()
    poses_np = np.concatenate([pos_0_np, pos_1_np], axis=0)
    if self.display is None:
      nodes, must_be[3] = pos_0_np.shape
      atomic_nums = 6*np.ones(2*nodes, dtype=int)
      atomic_nums[:nodes] = 5
      self.display = launch_atom_display(atomic_nums, poses_np)
      print("display launched!")
    else:
      print("about to update the display")
      input(">")
      self.display.update_pos(poses_np)
    return pos_1



if __name__ == "__main__":
  # TESTING
  tracker = TGTracker("Test Example Tracker")
  for offset in range(10):
    x = offset + 3*torch.randn(1000, requires_grad=True)
    x = tracker(x)
    loss = (x + x*x).sum()
    loss.backward()
  TGTracker.showall()
