import requests
import datetime
import torch
from PolygonRequestFormatter import PolygonRequestFormatter

class TrainingDataRequester():
  @staticmethod
  def getData(tick = "AAPL"):
    tday = datetime.date.today()
    td = datetime.timedelta(1)
    tdl = datetime.timedelta(730)
    endDay = tday - td
    startDay = tday-tdl
    data = requests.get(PolygonRequestFormatter.getRequestURL(tick, "day", startDay, endDay))
    data = data.json()
    results = data["results"]

    res = []
    for i in range(len(results)):
      x = results[i]
      res.append([x["c"], x["h"], x["l"], x["o"], x["v"]])

    return TrainingDataRequester.parseData(res)
    
  
  @staticmethod
  def parseData(arr):
    X = []
    y = []
    if (len(arr) > 110):
      for i in range(110, len(arr)):
        # create a new set of training data

        # create X inputs
        X_instance = []
        for j in range (i-110, i-10):
          X_instance.extend(arr[j])
        X.append(X_instance)

        # create Y answers
        init_y = arr[i-10][0]
        final_y = arr[i][0]
        change_percentage = (final_y - init_y) / init_y
        y_state = 0
        if (change_percentage < -0.2):
          y_state = 4
        elif (change_percentage < -0.05):
          y_state = 3
        elif (change_percentage < 0.05):
          y_state = 2
        elif (change_percentage < 0.2):
          y_state = 1
        else:
          y_state = 0
        y.append(y_state)

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    return[X, y]

