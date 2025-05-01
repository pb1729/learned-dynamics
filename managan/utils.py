import time
import torch


# SHAPE UTILS:

def prod(dims):
  ans = 1
  for dim in dims:
    ans *= dim
  return ans


class _MustBe:
  """ class for asserting that a dimension must have a certain value.
      the class itself is private, one should import a particular object,
      "must_be" in order to use the functionality. example code:
      `batch, chan, mustbe[32], mustbe[32] = image.shape`
      `*must_be[batch, 20, 20], chan = tens.shape` """
  def __setitem__(self, key, value):
    if isinstance(key, tuple):
      assert key == tuple(value), "must_be[%s] does not match dimension %s" % (str(key), str(value))
    else:
      assert key == value, "must_be[%d] does not match dimension %d" % (key, value)
must_be = _MustBe()


# TENSOR COMPARISON UTILS

def compare_tensors(t1, t2):
  def largest_elem(t):
    return abs(t).max().item()
  print("magnitude:", max(largest_elem(t1), largest_elem(t2)), "  difference:", largest_elem(t1 - t2))

def avg_relative_diff(a, b, show_as_plot=False):
  assert a.shape == b.shape
  y = (torch.abs(a - b)/torch.sqrt(0.0001 + 0.5*(a**2 + b**2))).mean(0).detach().cpu().numpy()
  if show_as_plot:
    import matplotlib.pyplot as plt
    plt.imshow(y)
    plt.show()
  return y.mean()

def typsz(x:torch.Tensor):
  return ((x**2).mean()**0.5).item()

# MODEL ACTIVATION SIZE PRINTER
def turn_on_actv_size_printing(module:torch.nn.Module):
  for name, mod in module.named_modules():
    depth = len(name.split("."))
    def fwd_hook(m, args, outs, curr_mod_name=name, curr_mod_depth=depth):
      def tostr(arg):
        if isinstance(arg, torch.Tensor):
          return str(typsz(arg))
        elif isinstance(arg, tuple):
          return f"({', '.join([tostr(subarg) for subarg in arg])})"
        else:
          return "_"
      if not isinstance(outs, tuple):
        outs = outs,
      arg_szs = ", ".join([
        tostr(arg)
        for arg in args
      ])
      out_szs = ", ".join([
        tostr(out)
        for out in outs
      ])
      print("  "*curr_mod_depth + f"{curr_mod_name}({arg_szs}) -> ({out_szs})")
    mod.register_forward_hook(fwd_hook)


# OTHER STUFF...

class PrintTiming:
  def __init__(self, start_message):
    self.start_message = start_message
  def __enter__(self):
    print(self.start_message)
    self.start_time = time.time()
    return self
  def __exit__(self, exc_type, exc_val, exc_tb):
    end_time = time.time()
    elapsed_time = end_time - self.start_time
    print(f"done after {elapsed_time:.4f} s")


class _GradRecord(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, callback):
    ctx.grad_rec_callback = callback
    return x
  @staticmethod
  def backward(ctx, grad_out):
    ctx.grad_rec_callback(grad_out)
    return grad_out, None
grad_record = _GradRecord.apply



# BATCHED EVALUATION:

def batched_xy_moment(x, y, batch=64000):
  """ compute product moment of two tensors in batches
      x: (instances, dim1)
      y: (instances, dim2)
      ans: (dim1, dim2)"""
  instances,          _ = x.shape
  must_be[instances], _ = y.shape
  ans = 0.
  for i in range(0, instances, batch):
    ans += torch.einsum("ix, iy -> xy", x[i:i+batch], y[i:i+batch])
  return ans/(instances - 1)

def batched_xy_moment_dot_3d(x, y, l=1, batch=64000):
  """ compute expected (outer product) square of tensor in batches
      tensor should contain irreps for the given choice of l
      takes dot product of the irreps in the tensor
      l: int -- characterizes which irrep we're dealing with
      x: (instances, dim1, 2l+1)
      y: (instances, dim2, 2l+1)
      ans: (dim1, dim2) """
  repdim = 2*l + 1
  instances,          _, must_be[repdim] = x.shape
  must_be[instances], _, must_be[repdim] = y.shape
  ans = 0.
  for i in range(0, instances, batch):
    ans += torch.einsum("ixv, iyv -> xy", x[i:i+batch], y[i:i+batch])
  return ans/(instances - 1)


def batched_model_eval(model, x, batch=16384):
  """ to avoid running out of memory, evaluate the model on a large tensor in batches
      Should only be called within torch.no_grad() context!
      model - the pytorch model to evaluate
      x     - the input we are feeding to the model. shape: (N, *shape)
      returns: the result of evaulating the model on x. shape: (N, *shape) """
  N = x.shape[0]
  ans = torch.zeros_like(x)
  for i in range(0, N, batch):
    ans[i:i+batch] = model(x[i:i+batch])
  return ans
