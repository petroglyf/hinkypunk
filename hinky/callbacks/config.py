""""Configuration for callbacks."""
from typing import Any, TypeVar

from pydantic import BaseModel, Field

OptionsConfigtype = TypeVar("OptionsConfigtype", bound=BaseModel)


class CallbackConfig(BaseModel):
  """Configuration for callbacks."""

  name: str = Field(description="Name of the callback.", min_length=3)
  class_name: str = Field(
    description="Fully qualified class name of the callback.", min_length=5,
  )
  every_n_epochs: int | None = Field(default=None, ge=1)
  every_n_steps: int | None = Field(default=None, ge=1)
  options: dict[str, Any] = Field(
    default={}, description="Additional options for the callback.",
  )
