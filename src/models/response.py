from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar('T')


class RestResponseList(BaseModel, Generic[T]):
    data: list[T]
    total: int
    offset: int
    limit: int
