from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    text: str = Field(min_length=1, max_length=1000)
    uppercase: bool | None = None
    mode: Literal['sync', 'async'] = 'sync'


class TextResponse(BaseModel):
    id: str
    text: str
