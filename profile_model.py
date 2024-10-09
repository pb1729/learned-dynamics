import torch
from torch.profiler import profile, record_function, ProfilerActivity

from config import get_predictor


def main(args):
  # load model
  predictor = get_predictor("model:" + args.fpath)
  # setup to make sure we're profiling a full training step
  model = predictor.model
  model.set_eval(False)
  # get input
  inp = torch.randn(args.batch, args.length, *predictor.shape, device=model.config.device)
  for i in range(args.burnin):
    print("burn-in %d" % i)
    model.train_step(inp)
  # setup profiling
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    with record_function("training_step"):
      model.train_step(inp)
  print("exporting...")
  prof.export_chrome_trace(args.saveto)
  print("done. wrote to %s" % args.saveto)
  print("view with: https://ui.perfetto.dev/")



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="profile_model")
  parser.add_argument("fpath")
  parser.add_argument("--batch", dest="batch", type=int, default=16)
  parser.add_argument("--length", dest="length", type=int, default=8)
  parser.add_argument("--burnin", dest="burnin", type=int, default=5)
  parser.add_argument("--saveto", dest="saveto", type=str, default="/tmp/chrome_trace.json")
  main(parser.parse_args())
