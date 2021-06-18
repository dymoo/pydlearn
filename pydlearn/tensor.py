import numpy as np
import device

class Tensor:
  def __init__(self, v):
    self.v = np.array(v, dtype=np.float32)

  # Custom initialisers
  @classmethod
  def uniform(self, shape):
    return Tensor(np.random.uniform(-1.0, 1.0, size=shape))

  @classmethod
  def ones(self, shape):
    return Tensor(np.ones(shape=shape))

  def zeroes(self, shape):
    return Tensor(np.zeros(shape=shape))

  # Properties
  @property
  def value(self):
    return self.v

  @property
  def shape(self):
    return self.v.shape

# Dynamically register accelerators
for name, val in device.__dict__.iteritems():
  print(name, val)