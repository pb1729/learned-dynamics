import torch


def prod(dims):
  ans = 1
  for dim in dims:
    ans *= dim
  return ans


def compare_tensors(t1, t2):
  def largest_elem(t):
    return abs(t).max().item()
  print("magnitude:", max(largest_elem(t1), largest_elem(t2)), "  difference:", largest_elem(t1 - t2))


class _MustBe:
  """ class for asserting that a dimension must have a certain value.
      the class itself is private, one should import a particular object,
      "must_be" in order to use the functionality. example code:
      `batch, chan, mustbe[32], mustbe[32] = image.shape` """
  def __setitem__(self, key, value):
    assert key == value, "must_be[%d] does not match dimension %d" % (key, value)
must_be = _MustBe()

# batched evaluation:

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


def batched_model_eval(model, input, outdim, batch=16384):
  """ to avoid running out of memory, evaluate the model on a large tensor in batches
      Should only be called within torch.no_grad() context!
      model  - the pytorch model to evaluate
      input  - the input we are feeding to the model. shape: (N, channels)
      outdim - integer, the size of the model's output
      returns: the result of evaulating the model. shape: (N, out_chan) """
  N, channels = input.shape
  ans = torch.zeros(N, outdim, device=input.device)
  for i in range(0, N, batch):
    ans[i:i+batch] = model(input[i:i+batch])
  return ans


