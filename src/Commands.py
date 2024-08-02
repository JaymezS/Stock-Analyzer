import math
from abc import abstractmethod
import Menu
from LinearStockPredictionModel import LongTermStockModelVersion1


type MenuMenu = Menu.Menu


class Command():
  @abstractmethod
  def execute(self) -> None:
    pass


class NullCommand(Command):
  def execute(self) -> None:
    return None


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
  def __init__(self, model: LongTermStockModelVersion1 | None) -> None:
    super().__init__()
    self.model = model
  
  def execute(self) -> None:
    if (self.model == None):
      print("-----------------------------------")
      print("WARNING: No Model available to save")
      print("-----------------------------------")
      return
    self.model.saveModel()
    print("----------------")
    print("   Model Saved  ")
    print("----------------")


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