from LinearStockPredictionModel import LongTermStockModelVersion1
from LinearStockPredictionModelTrainer import ModelTrainer
from TrainingDataRequester import TrainingDataRequester
from Menu import MenuProperties, MenuItem, Menu
from Commands import Command, DisplayMenuCommand, NullCommand, ExecuteMultipleCommandsCommand, SaveModelCommand


class Driver:
  def __init__(self):

    """
      Note: TrainingDataRequester's getData function returns the Input and output sets as 2 tensors that can be plugged directly into the model trainer's train x and train y
    """
    self.model: LongTermStockModelVersion1 | None = None


    self.mainMenu = Menu("Main Menu", "Model Trainer and Tester Menu")
    self.composeMainMenu()
    self.mainMenu.executeCommand()
  
  def composeMainMenu(self):

    trainModelMenuItem = self.composeTrainMenu()
    exitProgramItem = MenuItem("Exit Program", NullCommand())
    saveModelItem = MenuItem(
      "Save Model", 
      ExecuteMultipleCommandsCommand()
        .add_c(SaveModelCommand(self.model))
        .add_c(DisplayMenuCommand(self.mainMenu))
      )
    
    self.mainMenu.addItem(trainModelMenuItem).addItem(saveModelItem).addItem(exitProgramItem)

    return self.mainMenu

  def composeTrainMenu(self):
    trainModelMenu = Menu("Train Model Menu", "Browse Various ways to train your model")
    returnItem = MenuItem("Return To Main Menu", DisplayMenuCommand(self.mainMenu))
    trainModelMenu.addItem(returnItem)
    return trainModelMenu



Driver()