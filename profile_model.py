import torch
from torch.profiler import profile, record_function, ProfilerActivity

from config import get_predictor


def profile_torch(args, model, inp):
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    with record_function("training_step"):
      model.train_step(inp)
  print("exporting...")
  prof.export_chrome_trace(args.saveto)
  print("done. wrote to %s" % args.saveto)
  print("view with: https://ui.perfetto.dev/")


def profile_nsight(args, model, inp):
  """ https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59 """
  print("Profiling with NSIGHT, you should run using the following prefix:")
  print("nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -o my_profile python profile_model.py <your args>")
  torch.cuda.cudart().cudaProfilerStart()
  torch.cuda.nvtx.range_push("train_step")
  model.train_step(inp)
  torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()


def main(args):
  # load model
  predictor = get_predictor("model:" + args.fpath)
  # setup to make sure we're profiling a full training step
  model = predictor.model
  model.set_eval(False)
  # get input
  base_pred = predictor.get_base_predictor()
  state = predictor.sample_q(args.batch)
  inp = predictor.predict(args.length, state)
  for i in range(args.burnin):
    print("burn-in %d" % i)
    model.train_step(inp)
  # do profiling
  if args.nsight:
    profile_nsight(args, model, inp)
  else:
    profile_torch(args, model, inp)



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="profile_model")
  parser.add_argument("fpath")
  parser.add_argument("--batch", dest="batch", type=int, default=16)
  parser.add_argument("--length", dest="length", type=int, default=8)
  parser.add_argument("--burnin", dest="burnin", type=int, default=5)
  parser.add_argument("--saveto", dest="saveto", type=str, default="/tmp/chrome_trace.json")
  parser.add_argument("--nsight", dest="nsight", action="store_true")
  main(parser.parse_args())
