import numpy as np
import pyopencl as cl
from functools import lru_cache
from pydlearn.function import Function

def gpu_buffer(ctx, hostbuf=None, shape=None):
  return cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,\
    hostbuf=(np.zeros(shape) if shape else hostbuf))

@lru_cache
def gpu_kernel(code, reduce, gid_count, **kwargs):
  return (cl.Program('__kernel void op('+''.join(f'__global const float *{k}_g,' for k in kwargs)+\
    ' __global float '+('' if reduce else '*')+'res_g) { '+\
      '\n'.join(f'int gid_{gid} = get_global_id({gid});' for gid in range(gid_count))+code+';}').build()).op

def gpu_op(queue, code, reduce=False, gid_count=1, **kwargs):
  # TODO: this is not right!
  shape = (1,) if reduce else kwargs[list(kwargs.keys())[0]].shape

  buffers = {}
  for k in kwargs:
    buffers[k] = gpu_buffer(hostbuf=kwargs[k])
  buffers['res_buf'] = gpu_buffer(shape=shape)

  prg = gpu_kernel(code, reduce, gid_count, **kwargs)
  prg(queue, shape, None, **buffers)

  res = np.empty_like(shape)
  cl.enqueue_copy(queue, res, buffers['res_buf'])
  return res

class gpu:
  def __init__(self):
    self.cl_ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.cl_ctx)

  class sum(Function):
    def forward(ctx, v):
      return np.sum(v)

    def backward(ctx):
      return 0