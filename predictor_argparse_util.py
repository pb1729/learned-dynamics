from typing_extensions import List

from managan.config import get_predictor
from managan.predictor import Predictor, ModelPredictor, LongModelPredictor


def args_to_predictor_list(args):
  return model_list_to_predictor_list(args.models, args.overrides)

def model_list_to_predictor_list(models, overrides=None) -> List[Predictor]:
  if overrides is None:
    overrides = []
  if len(overrides) < len(models):
    overrides = [None for i in range(len(models) - len(overrides))] + overrides
  ans = []
  for model, override in zip(models, overrides):
    predictor:Predictor = get_predictor(model, override_base=override)
    if isinstance(predictor, ModelPredictor) or isinstance(predictor, LongModelPredictor): # add the base predictor for a model first.
      ans.append(predictor.get_base_predictor())
    ans.append(predictor)
  return ans

def conv_override(s):
  if s == "None": return None
  if s == ".":    return None
  return s

def add_model_list_arg(arg_parser):
  arg_parser.add_argument("-M", dest="models",    action="extend", nargs="+", type=str)
  arg_parser.add_argument("-O", dest="overrides", action="extend", nargs="+", type=conv_override)
