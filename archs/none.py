

class EmptyModel:
  def __init__(self, config):
    self.config = config
  @staticmethod
  def load_from_dict(states, config):
    return EmptyModel(config)
  @staticmethod
  def makenew(config):
    return EmptyModel(config)
  def save_to_dict(self):
    return {}

class EmptyTrainer:
  def __init__(self, model, board):
    pass
  def step(self, i, trajs):
    assert False, "not implemented"


# export model class and trainer class:
modelclass   = EmptyModel
trainerclass = EmptyTrainer



