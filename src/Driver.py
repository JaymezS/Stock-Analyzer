from LinearStockPredictionModel import LongTermStockModelVersion1
from LinearStockPredictionModelTrainer import ModelTrainer
from LinearStockPredictionModelTester import ModelTester
from TrainingDataRequester import TrainingDataRequester
from Menu import MenuProperties, MenuItem, Menu
import Commands


class Driver:
  def __init__(self):

    """
      Note: TrainingDataRequester's getData function returns the Input and output sets as 2 tensors that can be plugged directly into the model trainer's train x and train y
    """
    self.model: LongTermStockModelVersion1 = LongTermStockModelVersion1()
    self.trainer: ModelTrainer = ModelTrainer(self.model)
    self.tester: ModelTester = ModelTester(self.model)

    self.mainMenu = Menu("Main Menu", "Model Trainer and Tester Menu")
    self.composeMainMenu()
    self.mainMenu.executeCommand()
  
  def composeMainMenu(self):

    trainModelMenuItem = self.composeTrainMenu()
    exitProgramItem = MenuItem("Exit Program", Commands.NullCommand())
    saveModelItem = MenuItem(
      "Save the current Model", 
      Commands.ExecuteMultipleCommandsCommand()
        .add_c(Commands.SaveModelCommand(self.model))
        .add_c(Commands.DisplayMenuCommand(self.mainMenu))
      )
    loadModelItem = MenuItem(
      "Load the Saved Model",
      Commands.ExecuteMultipleCommandsCommand()
        .add_c(Commands.LoadModelCommand(self.model))
        .add_c(Commands.DisplayMenuCommand(self.mainMenu))
    )
    predictStockItem = MenuItem(
      "Predict the future of a stock", 
      Commands.ExecuteMultipleCommandsCommand()
        .add_c(Commands.PredictStockCommand(self.model))
        .add_c(Commands.DisplayMenuCommand(self.mainMenu))
    )
    
    (self.mainMenu
     .addItem(trainModelMenuItem)
     .addItem(saveModelItem)
     .addItem(loadModelItem)
     .addItem(predictStockItem)
     .addItem(exitProgramItem)
    )

    return self.mainMenu

  def composeTrainMenu(self):
    trainModelMenu = Menu("Training and Testing Menu", "Browse Various ways to train/test your model")
    trainByTicketItem = MenuItem("Train By Ticket", Commands.ExecuteMultipleCommandsCommand()           
      .add_c(Commands.TrainModelByTicketCommand(self.trainer)) 
      .add_c(Commands.DisplayMenuCommand(trainModelMenu))
    )
    testByTicketItem = MenuItem("Test By Ticket", Commands.ExecuteMultipleCommandsCommand()
      .add_c(Commands.TestModelByTicketCommand(self.tester))
      .add_c(Commands.DisplayMenuCommand(trainModelMenu))
    )
    returnItem = MenuItem("Return To Main Menu", Commands.DisplayMenuCommand(self.mainMenu))


    trainModelMenu.addItem(trainByTicketItem).addItem(testByTicketItem).addItem(returnItem)
    return trainModelMenu

Driver()