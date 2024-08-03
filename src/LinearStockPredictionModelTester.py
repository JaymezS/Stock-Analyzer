import torch.nn as nn
import torch.nn.functional as F
import torch
from TrainingDataRequester import TrainingDataRequester

class ModelTester():
  def __init__(self, model: nn.Module):
    self.model = model


  def test(self, ticker: str):
    
    data = TrainingDataRequester.getData(ticker)
    X = data[0]
    y = data[1]


    total_cases = len(X)
    wrong_cases = 0


    for i in range (total_cases):
      result = self.model.eval(X[i])

      guess = result.tolist().index(max(result))
      answer = y[i].tolist()

      if (not guess == answer):
        wrong_cases += 1
    print("________________________________________________________________________________")
    print(f"Out of {total_cases} cases, {wrong_cases} were wrong")
    print(f"Accuracy: {((total_cases-wrong_cases)/total_cases) * 100}% Accurate for stock: {ticker}")
    print("____________________________________________________________________________________")
