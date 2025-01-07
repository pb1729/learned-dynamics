import torch
import importlib

from .statefiles import DatasetPredictor
from .predictor import Predictor, ModelPredictor, get_sim_predictor, get_hoomd_predictor, get_openmm_predictor


ARCH_PREFIX = "archs."


def get_predictor(predictor_spec, override_base=None):
  """ given a specification for a predictor, construct the actual predictor.
      for backwards compatability, if no predictor type is specified, we return a SimPredictor """
  if ":" in predictor_spec:
    pred_type, rest = predictor_spec.split(":")
  else:
    pred_type, rest = "sim", predictor_spec
  if "sim" == pred_type:
    ans = get_sim_predictor(rest)
  elif "hoomd" == pred_type:
    ans = get_hoomd_predictor(rest)
  elif "openmm" == pred_type:
    ans = get_openmm_predictor(rest)
  elif "dataset" == pred_type:
    ans = DatasetPredictor(rest)
  elif "model" == pred_type: # treat "rest" as a path
    ans = ModelPredictor(load(rest, override_base))
  else: assert False, "unknown predictor type"
  ans.name = ":".join([pred_type, rest]) # tag predictor with a name for convenience
  return ans


class Config:
  """ configuration class for training runs """
  def __init__(self, pred_spec, arch_name,
               device="cuda", batch=1, simlen=16, nsteps=65536, save_every=512,
               arch_specific=None, trained_for=0, mutations=None):
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
    self.mutations = [] if mutations is None else mutations
    self.trained_for = trained_for
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
        "trained_for": self.trained_for,
        "mutations": self.mutations,
      }
    return args, kwargs
  def add_mutation(self, mut_desc):
    self.mutations.append(f"TRAIN FOR: {self.trained_for} steps")
    self.mutations.append(f"MUTATION: {mut_desc}")
    self.trained_for = 0
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
    for mutation in self.mutations:
      ans.append(mutation)
    ans.append(f"TRAIN FOR: {self.trained_for} steps")
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

def load(path, override_base=None):
  data = torch.load(path, weights_only=False)
  pred_spec = data["args"][0] if override_base is None else override_base
  config = Config(pred_spec, *data["args"][1:], **data["kwargs"])
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
