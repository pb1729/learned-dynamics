import datetime

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from managan.config import get_predictor, makenew, load_config
from managan.predictor import ModelPredictor



def do_op_to_be_profiled(args, model, inp):
  if args.inference:
    model.predict(inp)
  else:
    model.train_step(inp)


def profile_torch(args, model, inp):
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    with record_function("training_step"):
      do_op_to_be_profiled(args, model, inp)
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
  do_op_to_be_profiled(args, model, inp)
  torch.cuda.nvtx.range_pop()
  torch.cuda.cudart().cudaProfilerStop()


def main(args):
  if args.newmodel:
    config = load_config(args.fpath)
    model = makenew(config)
    predictor = ModelPredictor(model)
  else:
    # load model
    predictor = get_predictor("model:" + args.fpath, override_base=args.override)
    model = predictor.model
  model.set_eval(False)
  # fill in default save location if needed
  if args.saveto is None:
    currtime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    modelnm = model.config.arch_name
    args.saveto = f"traces/{currtime}_{modelnm}.json"
  # get input
  state = predictor.sample_q(model.config.batch if args.batch is None else args.batch)
  inp = predictor.predict(model.config.simlen if args.length is None else args.length, state)
  for i in range(args.burnin):
    print("burn-in %d" % i)
    do_op_to_be_profiled(args, model, inp)
  # do profiling
  if args.nsight:
    profile_nsight(args, model, inp)
  else:
    profile_torch(args, model, inp)



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="profile_model")
  parser.add_argument("fpath")
  parser.add_argument("--batch", dest="batch", type=int, default=None)
  parser.add_argument("--length", dest="length", type=int, default=None)
  parser.add_argument("--burnin", dest="burnin", type=int, default=5)
  parser.add_argument("--saveto", dest="saveto", type=str, default=None)
  parser.add_argument("--override", dest="override", type=str, default=None)
  parser.add_argument("--inference", dest="inference", action="store_true")
  parser.add_argument("--nsight", dest="nsight", action="store_true")
  parser.add_argument("--newmodel", dest="newmodel", action="store_true")
  main(parser.parse_args())
