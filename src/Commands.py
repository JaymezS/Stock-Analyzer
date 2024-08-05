from __future__ import annotations

import torch
import math
from abc import abstractmethod
import Menu
import LinearStockPredictionModel
from TrainingDataRequester import TrainingDataRequester
from HundredDayDataRequester import HundredDayDataRequester


type MenuMenu = Menu.Menu


class Command():
  @abstractmethod
  def execute(self) -> None:
    pass


class NullCommand(Command):
  def execute(self) -> None:
    return None


class DebugCommand(Command):
  def __init__(self, print: str) -> None:
    super().__init__()
    self.print = print

  def execute(self) -> None:
    print(self.print)


class DisplayMenuCommand(Command):
  def __init__(self, menu: MenuMenu) -> None:
    super().__init__()
    self.menu = menu

  def execute(self):
    print("///////////////////////////////////////")
    print("---------------------------------------")
    print(self.menu.title)
    print(self.menu.description)

    print("------------------------------------------")
    print("Choose an action below:")
    for i in range(len(self.menu.items)):
      print(f"{i+1}: {self.menu.items[i].title}")

    self.choose()
  
  def choose(self):
    user_input = 0
    while (True):
      user_input = int(input("Input the number of your choice: "))
      if (math.isnan(user_input)):
        print("Input is not a valid number, please try again. ")
      elif (user_input < 1 or user_input > len(self.menu.items)):
        print("That is not a valid option, please try again. ")
      else:
        user_input = user_input - 1
        break
    
    self.menu.items[user_input].executeCommand()
    

class SaveModelCommand(Command):
  def __init__(self, model: LinearStockPredictionModel.LongTermStockModelVersion1) -> None:
    super().__init__()
    self.model = model
  
  def execute(self) -> None:
    name: str = input("Input the file name to save the model: ").strip()
    self.model.saveModel(name)
    print("----------------")
    print("   Model Saved  ")
    print("----------------")


class LoadModelCommand(Command):
  def __init__(self, model: LinearStockPredictionModel.LongTermStockModelVersion1) -> None:
    super().__init__()
    self.model: LinearStockPredictionModel.LongTermStockModelVersion1 = model
  
  def execute(self) -> None:
    name: str = input("Input the name of the model to load (without '.pt'): ").strip()
    self.model.loadModel(name)


class TrainModelByTicketCommand(Command):
  def __init__(self, trainer) -> None:
    super().__init__()
    self.trainer = trainer
    

  def execute(self) -> None:
    t = input("Input the ticket to train on (eg. AAPL): ").strip()
    n = int(input("Input the number of training cycle/epochs: "))
    data = TrainingDataRequester.getData(t)
    if (data == None):
      print("cannot train on this stock")
      return
    X = data[0]
    y = data[1]
    if (math.isnan(n)):
      self.trainer.train(X, y)
    else:
      self.trainer.train(X, y, n)


class SPTrainCommand(Command):
  def __init__(self, trainer) -> None:
    super().__init__()
    self.trainer = trainer
  
  def execute(self) -> None:
    n = int(input("Input the number of training cycle/epochs: "))
    self.trainer.SP_train(n)


class TestModelByTicketCommand(Command):
  def __init__(self, tester) -> None:
    super().__init__()
    self.tester = tester

  def execute(self) -> None:
    t = input("Input the ticket to test on (eg. AAPL): ").strip()
    self.tester.test(t)


class PredictStockCommand(Command):
  def __init__(self, model: LinearStockPredictionModel.LongTermStockModelVersion1) -> None:
    super().__init__()
    self.model = model

  def execute(self):
    t = input("Input the ticket to predict (eg. AAPL): ").strip()
    data = HundredDayDataRequester().getData(t)
    tensor = []
    for i in data:
      tensor.extend(i)
    print(len(tensor))
    tensor = torch.FloatTensor(tensor)
    ans = self.model.eval(tensor)


    res = ["grow by over 20%", "grow by 5~20%", "change by -5~5%", "fall by 5~20%", "fall by more than 20%"]
    ans = ans.tolist().index(max(ans))

    print(f"Result: The stock's value per share will {res[ans]} in the next 10 days")
    print("Prediction is made on the stock's previous 100 day data, and may not be accurate")




class ExecuteMultipleCommandsCommand(Command):
  def __init__(self) -> None:
    super().__init__()
    self.commands: list[Command] = []
  
  def add_c(self, c: Command):
    self.commands.append(c)
    return self

  def execute(self):
    for command in self.commands:
      command.execute()