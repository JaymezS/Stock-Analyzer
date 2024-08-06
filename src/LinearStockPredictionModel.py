import torch
import torch.nn as nn
import torch.nn.functional as F


# take past 100 day's daily market values as inputs, each with volume, open/closing value, and highest/lowest (o/c, h/l, v)
# output 5 possibilities based on the results of the stock after 100 days 
# strong buy (20%, inf], buy (5-20%], hold(-5-5%], sell (-20%-(-5%)], strong sell (-inf, -20%]
# string buy = 1, strong sell = 5
class LongTermStockModelVersion1(nn.Module):

  def __init__(self, input_layer: int = 500, out: int = 42) -> None:
    super().__init__()

    self.fc1 = nn.Linear(input_layer, 1000)
    self.fc2 = nn.Linear(1000, 900)
    self.fc3 = nn.Linear(900, 800)
    self.fc4 = nn.Linear(800, 700)
    self.fc5 = nn.Linear(700, 600)
    self.fc6 = nn.Linear(600, 500)
    self.fc7 = nn.Linear(500, 400)
    self.fc8 = nn.Linear(400, 300)
    self.fc9 = nn.Linear(300, 200)
    self.fc10 = nn.Linear(200, 100)
    self.fc11 = nn.Linear(100, 50)
    self.out = nn.Linear(50, out)

  def forward(self, input):
    x = input
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))
    x = F.relu(self.fc7(x))
    x = F.relu(self.fc8(x))
    x = F.relu(self.fc9(x))
    x = F.relu(self.fc10(x))
    x = F.relu(self.fc11(x))
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

