import device
import pickle

class Network:
  ctx = []

  def __init__(self, device=device.gpu):
    self.device = device

  def backward(self):
    return 0

  def save(self, path):
    with open(path, 'wb') as f:
      pickle.dump(self, f)

  @staticmethod
  def load(path):
    with open(path, 'rb') as f:
      return pickle.load(f)