import torch
import importlib

from sims import sims
from hoomd_sims import hoomd_sims
from openmm_sims import openmm_sims
from predictor import Predictor, ModelPredictor, SimPredictor, HoomdPredictor, OpenMMPredictor


ARCH_PREFIX = "archs."


def get_predictor(predictor_spec):
  """ given a specification for a predictor, construct the actual predictor.
      for backwards compatability, if no predictor type is specified, we return a SimPredictor """
  if ":" in predictor_spec:
    pred_type, rest = predictor_spec.split(":")
  else:
    pred_type, rest = "sim", predictor_spec
  if "sim" == pred_type:
    ans = SimPredictor(sims[rest])
  elif "hoomd" == pred_type:
    ans = HoomdPredictor(hoomd_sims[rest])
  elif "openmm" == pred_type:
    ans = OpenMMPredictor(openmm_sims[rest])
  elif "model" == pred_type: # treat "rest" as a path
    ans = ModelPredictor(load(rest))
  else: assert False, "unknown predictor type"
  ans.name = ":".join([pred_type, rest]) # tag predictor with a name for convenience
  return ans


class Config:
  """ configuration class for training runs """
  def __init__(self, pred_spec, arch_name,
               device="cuda", batch=1, simlen=16, nsteps=65536, save_every=512,
               arch_specific=None, **kwargs_rest):
    self.pred_spec = pred_spec
    self.arch_name = arch_name
    self.predictor:Predictor = get_predictor(pred_spec)
    self.device = device
    self.batch = batch
    self.simlen = simlen
    self.nsteps = nsteps
    self.save_every = save_every
    if isinstance(nsteps, list):
      assert all(map(lambda ns: ns % save_every == 0, nsteps))
    else:
      assert nsteps % save_every == 0
    self.modelclass, self.trainerclass = self.get_model_and_trainer_classes()
    if arch_specific is None: arch_specific={}
    self.arch_specific = arch_specific # should be a dictionary of strings and ints
  def get_model_and_trainer_classes(self):
    arch_module = importlib.import_module(ARCH_PREFIX + self.arch_name)
    return arch_module.modelclass, arch_module.trainerclass
  def get_args_and_kwargs(self):
    """ find the args and kwargs that one would need to reconstruct this config """
    args = [self.pred_spec, self.arch_name]
    kwargs = {
        "device": self.device,
        "batch": self.batch,
        "simlen": self.simlen,
        "nsteps": self.nsteps,
        "save_every": self.save_every,
        "arch_specific": self.arch_specific,
      }
    return args, kwargs
  def __str__(self):
    args, kwargs = self.get_args_and_kwargs()
    ans = ["Config( \"%s\" ; \"%s\" ) {" % tuple(args)]
    for kw in kwargs:
      if kw != "arch_specific":
        ans.append("  %s: %s" % (kw, str(kwargs[kw])))
    ans.append("  arch_specific: {")
    for key in kwargs["arch_specific"]:
      ans.append("    %s: %s" % (key, str(kwargs["arch_specific"][key])))
    ans.append("  }")
    ans.append("}")
    return "\n".join(ans)
  def __getitem__(self, key):
    """ allows architectures to access arch_specific configuration using the indexing operator """
    return self.arch_specific[key]
  def __setitem__(self, key, value):
    self.arch_specific[key] = value




# universal functions for models. models must have the following behaviours:
# * .config                         is the config for that model
# * @staticmethod .load_from_dict() creates the model from a dict of states
# * .save_to_dict()                 records the model states into a dictionary
# * @staticmethod .makenew()        creates a new instance of the model

def load_config(path):
  data = torch.load(path, weights_only=False)
  return Config(*data["args"], **data["kwargs"])

def load(path):
  data = torch.load(path, weights_only=False)
  config = Config(*data["args"], **data["kwargs"])
  return config.modelclass.load_from_dict(data["states"], config)

def save(model, path):
  config_args, config_kwargs = model.config.get_args_and_kwargs()
  states = model.save_to_dict()
  torch.save({
      "args": config_args,
      "kwargs": config_kwargs,
      "states": states,
    }, path)

def makenew(config):
  return config.modelclass.makenew(config)
