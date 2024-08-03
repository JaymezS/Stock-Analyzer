import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy

class ModelTrainer():
  def __init__(self, model: nn.Module):
    self.model = model
    self.epochs = 10000
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
    self.losses = []


  def train(self, train_x, train_y):
    for i in range(self.epochs):
      y_pred = self.model.forward(train_x)
      loss = self.criterion(y_pred, train_y)
      self.losses.append(loss.detach().numpy())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()