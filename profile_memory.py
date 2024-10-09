import torch
from collections import defaultdict
import traceback
import matplotlib.pyplot as plt

from config import get_predictor
from predictor import ModelPredictor


class MemoryProfiler:
  """ Usage: if profiler is an instance of this object, then setup profiling like so:
      profiler.register_global_hooks()
      with torch.autograd.graph.saved_tensors_hooks(*profiler.get_pack_hooks()):
        ... # do your operation to be profiled here
  """
  def __init__(self):
    self.curr_module = ["nothing"]
    self.module_totals = defaultdict(lambda: 0)
    self.module_counts = defaultdict(lambda: 0)
  def register_global_hooks(self):
    def pre_fwd_hook(module, inp):
      self.curr_module.append(module.__class__.__name__)
    def fwd_hook(module, inp, out):
      module_nm = self.curr_module.pop()
      assert module_nm == module.__class__.__name__, "tried to pop a module frame that does not match the corresponding pushed frame"
    torch.nn.modules.module.register_module_forward_pre_hook(pre_fwd_hook)
    torch.nn.modules.module.register_module_forward_hook(fwd_hook)
  def get_pack_hooks(self):
    def pack_hook(x):
      self.record_mem_usage(x)
      return x
    def unpack_hook(x):
      return x
    return pack_hook, unpack_hook
  def record_mem_usage(self, x:torch.Tensor):
    dtype = x.dtype
    size = x.numel()
    usage = size*dtype.itemsize
    self.module_totals[self.curr_module[-1]] += usage
    self.module_counts[self.curr_module[-1]] += 1


def main(args):
  # load model
  predictor:ModelPredictor = get_predictor("model:" + args.fpath)
  # setup to make sure we're profiling a full training step
  model = predictor.model
  model.set_eval(False)
  # get input
  inp = torch.randn(args.batch, args.length, *predictor.shape, device=model.config.device)
  # setup profiling
  profiler = MemoryProfiler()
  profiler.register_global_hooks()
  with torch.autograd.graph.saved_tensors_hooks(*profiler.get_pack_hooks()):
    model.train_step(inp)
  # plotting and results
  x, y = [], []
  for key in profiler.module_totals:
    total, count = profiler.module_totals[key], profiler.module_counts[key]
    avg = total/count
    print(f"{key.rjust(32, ' ')}: {total: 12d} = {count: 8d} * {avg: 8f}")
    x.append(key)
    y.append(total/1e6)
  plt.bar(x, y)
  plt.xticks(rotation=90)
  plt.ylabel("memory usage [Mb]")
  plt.show()


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="profile_memory")
  parser.add_argument("fpath")
  parser.add_argument("--batch", dest="batch", type=int, default=1)
  parser.add_argument("--length", dest="length", type=int, default=8)
  main(parser.parse_args())
