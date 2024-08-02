from Commands import Command, DisplayMenuCommand

class MenuProperties():
  def __init__(self, title: str):
    self.title = title
    self.command = None
  
  def assignCommand(self, c: Command):
    self.command = c
  
  def executeCommand(self):
    self.command.execute()


class MenuItem(MenuProperties): 
  def __init__(self, title: str, command: Command):
    super().__init__(title)
    self.assignCommand(command)


class Menu(MenuProperties):

  def __init__(self, title: str, description: str):
    super().__init__(title)
    self.description = description
    self.assignCommand(DisplayMenuCommand(self))
    self.items: list[MenuProperties] = []
    
  def addItem(self, item: MenuProperties):
    self.items.append(item)
    return self