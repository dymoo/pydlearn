from pydlearn import Network, Tensor

import tqdm

# Simple mnist network consisting of 3 layers
# dot(32) -> tanh -> dot(64) -> tanh -> dot(10) -> softmax
class mnist_simple(Network):
  def __init__(self):
    l1 = Tensor.uniform(32)
    # l2 = tensor.Tensor.uniform(64)
    # l_out = tensor.Tensor.uniform(10)

  def forward(self, x):
    x = x.dot(self.l1)
    print(x)
    # x = x.dot(self.l1).tanh()
    # x = x.dot(self.l2).tanh()
    # return x.dot(self.l_out).softmax()

EPOCHS = 10

nn = mnist_simple()
for epoch in tqdm(range(EPOCHS)):
  from time import sleep
  sleep(1)
  print('epoch!')

save = input("save? (y/n)")
if save == "y":
  nn.save('mnist_simple.pydlnn')
  print('saved!')