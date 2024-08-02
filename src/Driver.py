from LinearStockPredictionModel import LongTermStockModelVersion1
from LinearStockPredictionModelTrainer import ModelTrainer
from TrainingDataRequester import TrainingDataRequester
from Menu import MenuProperties, MenuItem, Menu
import Commands


class Driver:
  def __init__(self):

    """
      Note: TrainingDataRequester's getData function returns the Input and output sets as 2 tensors that can be plugged directly into the model trainer's train x and train y
    """
    self.model: LongTermStockModelVersion1 = LongTermStockModelVersion1()


    self.mainMenu = Menu("Main Menu", "Model Trainer and Tester Menu")
    self.composeMainMenu()
    self.mainMenu.executeCommand()
  
  def composeMainMenu(self):

    trainModelMenuItem = self.composeTrainMenu()
    exitProgramItem = MenuItem("Exit Program", Commands.NullCommand())
    saveModelItem = MenuItem(
      "Save Model", 
      Commands.ExecuteMultipleCommandsCommand()
        .add_c(Commands.SaveModelCommand(self.model))
        .add_c(Commands.DisplayMenuCommand(self.mainMenu))
      )
    loadModelItem = MenuItem(
      "Load A New Model",
      Commands.ExecuteMultipleCommandsCommand()
        .add_c(Commands.LoadModelCommand(self.model))
        .add_c(Commands.DisplayMenuCommand(self.mainMenu))
    )
    
    self.mainMenu.addItem(trainModelMenuItem).addItem(saveModelItem).addItem(loadModelItem).addItem(exitProgramItem)

    return self.mainMenu

  def composeTrainMenu(self):
    trainModelMenu = Menu("Train Model Menu", "Browse Various ways to train your model")
    returnItem = MenuItem("Return To Main Menu", Commands.DisplayMenuCommand(self.mainMenu))
    trainModelMenu.addItem(returnItem)
    return trainModelMenu



Driver()