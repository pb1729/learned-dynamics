# Tensorboard Visualization Code
#
#   --- Utilization: ---
#
# tensorboard --logdir=runs
# ssh -NfL 6006:localhost:6006 username@yourmachine.example.com
# firefox http://localhost:6006


LOG_DIR = "runs"


try:
  from torch.utils.tensorboard import SummaryWriter
  import torchvision
except ModuleNotFoundError:
  print("Could not import tensorboard or torchvision. Falling back to dummy class definition.")
  class TensorBoard:
    """ dummy tensorboard that implements the same methods as a real one """
    def __init__(self, name):
      self.name = name
    def img_grid(self, label, images):
      pass
    def scalar(self, label, i, val):
      pass
else:
  class TensorBoard:
    """ logs various kinds of data to a tensor board """
    def __init__(self, name):
      self.name = name
      self.writer = SummaryWriter("/".join([LOG_DIR, self.name]))
      self.histories = {}
    def img_grid(self, label, images):
      grid = torchvision.utils.make_grid(images)
      self.writer.add_image(label, grid)
    def scalar(self, label, i, val):
      self.writer.add_scalar(label, val, i)
