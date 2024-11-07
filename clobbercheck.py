import numpy as np
import torch
import matplotlib.pyplot as plt


# constants
MEM_MARGIN = 10
FLOAT_SZ = 4


class ClobberChecker:
  def __init__(self, array_sizes):
    self.array_sizes = array_sizes
  def __enter__(self):
    self.tensors = []
    for sz in self.array_sizes:
      self.tensors.append(torch.ones(sz, dtype=torch.int8, device="cuda"))
    return self
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.tensors.clear()
  def report(self):
    clobber_count = 0
    for x in self.tensors:
      clobbers = (x != 1)
      clobber_count += clobbers.sum()
    print(f"GPU Clobber count: {clobber_count}")

class CPUClobberChecker:
  def __init__(self, array_sizes):
    self.array_sizes = array_sizes
  def __enter__(self):
    self.arrays = []
    for sz in self.array_sizes:
      self.arrays.append(np.ones(sz, dtype=np.int8))
    return self
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.report()
    self.arrays.clear()
  def report(self):
    clobber_count = 0
    for x in self.arrays:
      clobbers = (x != 1)
      clobber_count += clobbers.sum()
    print(f"CPU Clobber count: {clobber_count}")
