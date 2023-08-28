from pydantic import BaseModel


class CandlePredictionBody(BaseModel):
    model: str
    interval: str
