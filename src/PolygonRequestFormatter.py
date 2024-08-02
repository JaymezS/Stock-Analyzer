import os
from dotenv import load_dotenv

load_dotenv()


POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


class PolygonRequestFormatter():
  def __init__(self):
    pass

  @staticmethod
  def getRequestURL(ticker: str, timespan: str, fromDay: str, toDay: str) -> str:
    return f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/{timespan.lower()}/{fromDay}/{toDay}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
