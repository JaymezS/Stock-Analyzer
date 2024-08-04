import torch
import torch.nn as nn
import torch.nn.functional as F


# take past 100 day's daily market values as inputs, each with volume, open/closing value, and highest/lowest (o/c, h/l, v)
# output 5 possibilities based on the results of the stock after 100 days 
# strong buy (20%, inf], buy (5-20%], hold(-5-5%], sell (-20%-(-5%)], strong sell (-inf, -20%]
# string buy = 1, strong sell = 5
class LongTermStockModelVersion1(nn.Module):

  def __init__(self, input_layer: int = 500, l1: int = 5000, l2: int = 500, l3: int = 100, out: int = 5) -> None:
    super().__init__()
    self.fc1 = nn.Linear(input_layer, l1)
    self.fc2 = nn.Linear(l1, l2)
    self.fc3 = nn.Linear(l2, l3)
    self.out = nn.Linear(l3, out)

  def forward(self, input):
    x = input
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.out(x)
    return x
  

  def saveModel(self, name: str = "default_model_save"):
    torch.save(self.state_dict(), f"models/{name}.pt")

  def loadModel(self, name: str = "default_model_save"):
    self.load_state_dict(torch.load(f"models/{name}.pt"))

  def eval(self, input):
    with torch.no_grad():
      y_eval = self.forward(input)
    return y_eval

