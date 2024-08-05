import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy
import json
from TrainingDataRequester import TrainingDataRequester

class ModelTrainer():
  def __init__(self, model: nn.Module):
    self.model = model
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
    self.losses = []


  def train(self, train_x, train_y, epoch = 1000):
    for i in range(epoch):
      y_pred = self.model.forward(train_x)
      loss = self.criterion(y_pred, train_y)
      self.losses.append(loss.detach().numpy())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
  
  def SP_train(self, epoch = 10):
    f = open("assets/data/SPComponents.json")
    companies = json.load(f)["companies"]
    for i in range(epoch):
      for company in companies:
        d = TrainingDataRequester.getData(company)
        if d == None:
          continue
        X = d[0]
        y = d[1]
        self.train(X, y, epoch)



