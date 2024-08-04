import requests
import datetime
from PolygonRequestFormatter import PolygonRequestFormatter


class HundredDayDataRequester():
  @staticmethod
  def getData(tick: str):


    endDay = datetime.date.today() - datetime.timedelta(days=1)
    tdelta = datetime.timedelta(days=200)

    startDay = endDay - tdelta
    data = requests.get(PolygonRequestFormatter.getRequestURL(tick, "day", startDay, endDay))
    data = data.json()
    results = data["results"][-100:]


    res = []
    for i in range(len(results)):
      x = results[i]
      res.append([x["c"], x["h"], x["l"], x["o"], x["v"]])

    return res




