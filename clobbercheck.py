import torch
import matplotlib.pyplot as plt


# constants
MEM_MARGIN = 10
FLOAT_SZ = 4


class ClobberChecker:
  def __init__(self, chunk_sz=1_000_000):
    self.chunk_sz = chunk_sz
  def __enter__(self):
    self.tensors = []
    free_mem, _ = torch.cuda.mem_get_info()
    used_mem = 0
    while free_mem - used_mem > MEM_MARGIN*FLOAT_SZ*self.chunk_sz:
      self.tensors.append(torch.zeros(self.chunk_sz, device="cuda"))
      used_mem += FLOAT_SZ*self.chunk_sz
    return self
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.tensors.clear()
  def report(self):
    clobber_count = 0
    for x in self.tensors:
      clobbers = (x != 0)
      clobber_count += clobbers.to(torch.int32).sum()
    print(f"Clobber count: {clobber_count}")
