import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from polymer_util import rouse


# BASIS TRANSFORMS FOR PLOTTING:

def basis_transform_coords(x):
  """ x: (batch, poly_len, space_dim) """
  return x

def basis_transform_rouse(x):
  """ x: (batch, poly_len, space_dim) """
  _, poly_len, _ = x.shape
  W = np.stack([rouse(n, poly_len) for n in range(poly_len)])
  return np.einsum("nj, bjk -> bnk", W, x)

def basis_transform_neighbours(x):
  """ x: (batch, poly_len, space_dim) """
  _, poly_len, _ = x.shape
  n = np.arange(poly_len)
  return x[:, n] - x[:, (n+1)%poly_len]




# PLOTTING CODE:

def squarish_factorize(n):
  """ plotting utility function, gives as close to a square factorization of n as possible.
      returns integers x, y with x >= y """
  best_y = 1
  for y in range(2, int(np.sqrt(n)) + 1):
    if n % y == 0:
      best_y = y
  return n//best_y, best_y


class Plotter:
  def __init__(self, hist_transform=basis_transform_coords, flash_plot=False, samples_subset_size=20, title=None):
    self.hist_transform = hist_transform
    self.flash_plot = flash_plot
    self.samples_subset_size = samples_subset_size
    self.fig = None
    self.axes = None
    self.col_idx = 0
    self.title = title
  def _init_hist(self, poly_len, space_dim):
    self.fig = plt.figure(figsize=(20, 12))
    plt_w, plt_h = squarish_factorize(poly_len*space_dim)
    self.axes = [self.fig.add_subplot(plt_h, plt_w, n + 1) for n in range(poly_len*space_dim)]
    self.hist_ranges = []
  def plot_hist(self, samples):
    """ samples: (batch, poly_len, space_dim) """
    batch, poly_len, space_dim = samples.shape
    if self.fig is None: self._init_hist(poly_len, space_dim)
    samples = self.hist_transform(samples)
    for i in range(poly_len):
      hist_range = (np.min(samples[:, i]), np.max(samples[:, i]))
      if len(self.hist_ranges) <= i: # hist ranges will be filled in once, by the first call to this function
        self.hist_ranges.append((np.min(samples[:, i]), np.max(samples[:, i])))
      for j in range(space_dim):
        ax = self.axes[space_dim*i + j]
        ax.hist(samples[:, i, j], range=self.hist_ranges[i], bins=100, color=self._get_color(), alpha=0.7)
        ax.scatter([samples[:, i, j].mean()], [0.0], marker="d", facecolors=self._get_color(), edgecolors='black', s=[100], zorder=10)
    self.col_idx += 1
  def plot_hist_ic(self, ic):
    """ ic: (1, poly_len, space_dim) """
    _, poly_len, space_dim = ic.shape
    if self.fig is None: self._init_hist(poly_len, space_dim)
    samples = self.hist_transform(ic)
    for i in range(poly_len):
      for j in range(space_dim):
        ax = self.axes[space_dim*i + j]
        ax.scatter([samples[0, i, j]], [0.0], marker="d", facecolors="grey", edgecolors='black', s=[100], zorder=20)
        # make visual grouping of subplots obvious by supply a color cue:
        ax.set_facecolor(["#f0d0d0", "#d0f0d0", "#d0d0f0"][i % 3])
  def _init_samples(self, poly_len, space_dim):
    self.fig = plt.figure(figsize=(20, 12))
    if space_dim == 1:
      self.axes = [self.fig.add_subplot()]
    elif space_dim == 2 or space_dim == 3:
      self.axes = [self.fig.add_subplot(projection="3d")]
    else:
      raise NotImplementedError("4 dimensions and above not implemented")
  def plot_samples(self, samples):
    """ samples: (batch, poly_len, space_dim) """
    batch, poly_len, space_dim = samples.shape
    if self.fig is None: self._init_samples(poly_len, space_dim)
    for i in range(self.samples_subset_size):
      if i < batch:
        coords_xyz = [samples[i, :, j] for j in range(space_dim)]
        if space_dim == 1 or space_dim == 2:
          coords_xyz = [np.arange(poly_len)] + coords_xyz
        self.axes[0].plot(*coords_xyz, marker=".", color=self._get_color(), alpha=0.8)
    self.col_idx += 1
  def plot_samples_ic(self, ic):
    """ ic: (1, poly_len, 1) """
    batch, poly_len, space_dim = ic.shape
    if self.fig is None: self._init_samples(poly_len, space_dim)
    coords_xyz = [ic[0, :, j] for j in range(space_dim)]
    if space_dim == 1 or space_dim == 2:
      coords_xyz = [np.arange(poly_len)] + coords_xyz
    self.axes[0].plot(*coords_xyz, marker=".", color="black")
  def _get_color(self):
    return colormaps["tab10"](self.col_idx/9.5)
  def show(self):
    if self.title is not None:
      self.fig.suptitle(self.title)
    if self.flash_plot:
      self.fig.show(block=False)
      self.fig.pause(0.7)
      self.fig.close()
    else:
      plt.show()
    self._reset()
  def _reset(self):
    self.fig = None
    self.axes = None
    self.col_idx = 0




