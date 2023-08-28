from typing import Union

from pydantic import BaseModel


class TimeframePredictionBody(BaseModel):
    model: str
    interval: str
    indicatorTypes: list[str]
    rocLength: Union[int, None] = 9
    rsiLength: Union[int, None] = 14
