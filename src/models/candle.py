from pydantic import BaseModel


class Candle(BaseModel):
    binanceEventTimestamp: int
    symbol: str
    startIntervalTimestamp: int
    endIntervalTimestamp: int
    interval: str
    openPrice: str
    closePrice: str
    highPrice: str
    lowPrice: str
    volume: str
