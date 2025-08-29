# Fast, low-overhead tensor uniform 1d. We're not supposed to be messing with this stuff, but NVIDIA provided
# code (cuequivariance_ops_torch/tensor_product_uniform_1d_jit.py) with too much CPU overhead, so here we are.

from itertools import accumulate
from typing import Dict, List, Optional
from collections import namedtuple

import torch
from torch import nn

import cuequivariance as cueq
# import the fast cuda kernel stuffs, plus useful definitions
import cuequivariance_ops_torch._ext as ext

from managan.utils import must_be, prod


def map_dtype(jit, dtype):
  if dtype == torch.float32: return jit.Datatype.kFloat32
  raise ValueError(f"Unknown dtype {dtype}")

def map_buffer_dim(jit, o):
  if o == 0: return jit.Dimension.kScalar
  if o == 1: return jit.Dimension.kOneDimensional
  raise ValueError(f"Unknown dimension {o}")


class FastSegmentedPolynomialFromUniform1dJit(nn.Module):
  def __init__(self,
    polynomial: cueq.SegmentedPolynomial,
    name: str = "segmented_polynomial",
  ):
    super().__init__()

    operand_extent = None
    for o in polynomial.operands:
      torch._assert(
        o.ndim in [0, 1],
        "only 0 or 1 dimensional operands are supported by this method",
      )
      torch._assert(
        all(len(s) == o.ndim for s in o.segments),
        "all segments must have the same number of dimensions as the operand for this method",
      )
      torch._assert(
        o.all_same_segment_shape(),
        "all segments must have the same shape for this method",
      )
      if o.ndim == 1 and len(o.segments) > 0:
        if operand_extent is None:
          (operand_extent,) = o.segment_shape
        else:
          torch._assert(
            (operand_extent,) == o.segment_shape,
            "all operands must have the same extent for this method",
          )
    if operand_extent is None:
      operand_extent = 1

    for o, stp in polynomial.operations:
      torch._assert(
        stp.num_operands == len(o.buffers),
        "the number of operands must match the number of buffers",
      )
      torch._assert(
        stp.coefficient_subscripts == "", "the coefficients must be scalar"
      )

    self.num_inputs = polynomial.num_inputs
    self.num_outputs = polynomial.num_outputs
    self.name = name
    self.operand_extent = operand_extent
    self.buffer_dim = [o.ndim for o in polynomial.operands]
    torch._assert(
      all(buffer_dim in [0, 1] for buffer_dim in self.buffer_dim),
      "buffer dimensions must be 0 or 1",
    )
    self.buffer_num_segments = [len(o.segments) for o in polynomial.operands]
    self.dtypes = list(range(self.num_inputs)) + [-1]*polynomial.num_outputs
    self.num_operations = len(polynomial.operations)
    self.num_operands = [len(o.buffers) for o, stp in polynomial.operations]
    self.operations = [b for o, stp in polynomial.operations for b in o.buffers]
    self.num_paths = [stp.num_paths for o, stp in polynomial.operations]
    self.path_indices_start = [0] + list(
      accumulate([stp.num_paths * stp.num_operands for o, stp in polynomial.operations]))[:-1]
    self.path_coefficients_start = [0] + list(
      accumulate([stp.num_paths for o, stp in polynomial.operations]))[:-1]
    self.path_indices = [i
      for o, stp in polynomial.operations
      for p in stp.paths for i in p.indices]
    self.path_coefficients = [float(p.coefficients)
      for o, stp in polynomial.operations
      for p in stp.paths]
    self.math_dtype = torch.float32
    # precompute ops:
    operation_index = 0
    self.ops = []
    for i in range(self.num_operations):
      self.ops.append(self.operations[operation_index : operation_index + self.num_operands[i]])
      operation_index += self.num_operands[i]
    # construct the overall settings for this op:
    self.uniform_1d_settings = Uniform1dSettings(
      self.name, self.math_dtype, self.operand_extent, self.num_inputs, self.num_outputs, self.buffer_dim, self.buffer_num_segments,
      self.ops, self.num_paths, self.path_indices_start, self.path_coefficients_start, self.path_indices, self.path_coefficients)
  def forward(self, tensors: List[torch.Tensor]):
    # check shapes
    assert isinstance(tensors, list), "expected a list as input"
    rest = None
    for i in range(len(tensors)):
      assert tensors[i].dtype == self.math_dtype, "input has wrong dtype"
      segments_dim = self.operand_extent*self.buffer_num_segments[i]
      if rest is None:
        *rest, must_be[segments_dim] = tensors[i].shape
        batch_size = prod(rest)
      else:
        *must_be[rest], must_be[segments_dim] = tensors[i].shape
      tensors[i] = tensors[i].contiguous().reshape(batch_size, segments_dim)
    # apply kernel
    ans = Uniform1dJit.apply(self.uniform_1d_settings, batch_size, *tensors)
    return [
      tens.reshape(*rest, -1)
      for tens in ans
    ]


Uniform1dSettings = namedtuple("Uniform1dSettings", [
  "name", "math_dtype", "operand_extent", "num_inputs", "num_outputs", "buffer_dim", "buffer_num_segments", "ops",
  "num_paths", "path_indices_start", "path_coefficients_start", "path_indices", "path_coefficients"])



class Uniform1dJit(torch.autograd.Function):
  @staticmethod
  def forward(ctx, settings:Uniform1dSettings, batch_size:int, *tensors):
    assert len(tensors) == settings.num_inputs
    # save data for bwd
    ctx.save_for_backward(*tensors)
    ctx.settings = settings
    ctx.batch_size = batch_size
    # make outputs list
    outputs = []
    for i in range(settings.num_inputs, settings.num_inputs + settings.num_outputs):
      outputs.append(torch.empty(batch_size, settings.operand_extent*settings.buffer_num_segments[i],
        dtype=settings.math_dtype, device=tensors[0].device))
    # prep tensors
    tensors = [t.contiguous() for t in tensors]
    #tensors = [t.reshape(t.shape) for t in tensors] # this unwraps torch.Parameters into torch.Tensor
    tensors = tensors + outputs
    # prepare jit
    jit = ext.tensor_product_uniform_1d_jit
    # call jit
    total_num_ops = settings.num_inputs + settings.num_outputs
    jit.run(
      settings.name,
      map_dtype(jit, settings.math_dtype),
      settings.operand_extent,
      settings.num_inputs,
      settings.num_outputs,
      0, # no indexing
      [map_buffer_dim(jit, b) for b in settings.buffer_dim],
      settings.buffer_num_segments,
      [jit.BatchDimension.kBatched] * total_num_ops, # everything must be batched
      [-1] * total_num_ops, # no indexing
      [map_dtype(jit, t.dtype) for t in tensors],
      settings.ops,
      settings.num_paths,
      settings.path_indices_start,
      settings.path_coefficients_start,
      settings.path_indices,
      settings.path_coefficients,
      batch_size,
      tensors,
      torch.cuda.current_stream().cuda_stream
    )
    return tuple(outputs)
  @staticmethod
  def backward(ctx, *d_outputs):
    # unpack useful data saved to ctx
    *tensors, = ctx.saved_tensors
    settings:Uniform1dSettings = ctx.settings
    batch_size:int = ctx.batch_size
    needs_input_grad = ctx.needs_input_grad[2:] # slice off settings and batch_size args
    assert all(needs_input_grad), "holes in needs_input_grad not yet supported!"
    # compute the new values needed for some of the settings
    bwd_num_inputs = settings.num_inputs + settings.num_outputs
    bwd_num_outputs = sum(1 if ng else 0 for ng in needs_input_grad)
    bwd_buffer_dim = settings.buffer_dim + [
      settings.buffer_dim[idx]
      for idx, ng in enumerate(needs_input_grad) if ng
    ]
    bwd_buffer_num_segments = settings.buffer_num_segments + [
      settings.buffer_num_segments[idx]
      for idx, ng in enumerate(needs_input_grad) if ng
    ]

    # Compute a new list of ops! (see original code in tensor_product_uniform_1d_jit.py)
    bwd_ops = []
    bwd_num_paths = []
    bwd_path_indices_start = []
    bwd_path_coefficients_start = []
    output_idx = bwd_num_inputs
    for ng_idx, ng in enumerate(needs_input_grad):
      if not ng: continue
      # we want the derivative of input operand "idx"
      # and store it into output operand bwd_num_input_operands + output_idx
      # we have the gradients of the previous outputs in buffers orig_num_inputs ... orig_num_inputs + orig_num_outputs
      #   i.e. we can keep them as is!
      for ops_idx, op in enumerate(settings.ops):
        # for a given operation, if it uses "idx" at a position k:
        #   we replace "idx" at that position k with the output operand
        #   we replace the output operand with its gradient
        #   we add that to the list of operations
        #   we also have to replicate num_paths, num_indices_start, num_coefficients_start
        for op_idx, k in enumerate(op):
          if k == ng_idx:
            bwd_op = list(op)
            bwd_op[op_idx] = output_idx
            bwd_ops.append(bwd_op)
            bwd_num_paths.append(settings.num_paths[ops_idx])
            bwd_path_indices_start.append(settings.path_indices_start[ops_idx])
            bwd_path_coefficients_start.append(
              settings.path_coefficients_start[ops_idx]
            )
      output_idx += 1

    bwd_path_indices = settings.path_indices
    bwd_path_coefficients = settings.path_coefficients

    # Recursively call this autograd fn so we can support higher derivatives
    ans = Uniform1dJit.apply(
      Uniform1dSettings(settings.name + "_bwd",
        settings.math_dtype, settings.operand_extent, bwd_num_inputs, bwd_num_outputs, bwd_buffer_dim, bwd_buffer_num_segments,
        bwd_ops, bwd_num_paths, bwd_path_indices_start, bwd_path_coefficients_start, bwd_path_indices, bwd_path_coefficients),
      batch_size,
      *tensors, *d_outputs
    )
    return None, None, *ans
