# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Parameter utilities."""

import pathlib
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeAlias, TypedDict

import colorama
import numpy as np
from numpy import typing as npt
from typing_extensions import Self

import mujoco

Fore = colorama.Fore
Style = colorama.Style

ModifierFn: TypeAlias = Callable[[mujoco.MjSpec, Parameter], object]

_NestedFloats: TypeAlias = list[float] | list[_NestedFloats]

class _ParameterState(TypedDict):
  name: str
  nominal: _NestedFloats
  min_value: _NestedFloats
  max_value: _NestedFloats
  value: _NestedFloats
  frozen: bool

class InertiaType(Enum):
  Mass = 0
  MassIpos = 1
  Pseudo = 2

class Parameter:
  """A single (possibly multi-dimensional) parameter for system identification."""

  name: str
  nominal: npt.NDArray[np.float64]
  min_value: npt.NDArray[np.float64]
  max_value: npt.NDArray[np.float64]
  value: npt.NDArray[np.float64]
  frozen: bool
  modifier: ModifierFn | None
  inertia_type: InertiaType | None
  scale_rot_inertia: bool

  def __init__(
    self,
    name: str,
    nominal: float | npt.ArrayLike,
    min_value: float | npt.ArrayLike,
    max_value: float | npt.ArrayLike,
    frozen: bool = False,
    modifier: ModifierFn | None = None,
  ) -> None: ...
  @property
  def size(self) -> int: ...
  @property
  def shape(self) -> tuple[int, ...]: ...
  def apply_modifier(self, spec: mujoco.MjSpec) -> None:
    """Apply this parameter's modifier callback to *spec*, if one is set."""

  def as_vector(self) -> npt.NDArray[np.float64]:
    """Return the current value as a flat 1-D array."""

  def as_nominal_vector(self) -> npt.NDArray[np.float64]:
    """Return the nominal value as a flat 1-D array."""

  def update_from_vector(self, vector: npt.NDArray[np.float64]) -> None:
    """Update the current value from a flat vector."""

  def get_bounds(
    self,
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(lower, upper)`` bound arrays, each flat 1-D."""

  def reset(self) -> None:
    """Reset the current value to nominal."""

  def sample(
    self, rng: np.random.Generator | None = None
  ) -> npt.NDArray[np.float64]:
    """Sample a random value uniformly within bounds."""

  def move_off_bound(
    self, fraction: float = 0.05, at_bound_tol: float = 1e-3
  ) -> None:
    """Shift values at a bound into the interior of the bounds."""

  def __getstate__(self) -> _ParameterState: ...
  def __setstate__(self, state: _ParameterState) -> None: ...
  def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...

class ParameterDict:
  """An ordered collection of :class:`Parameter` objects."""

  parameters: dict[str, Parameter]

  def __init__(
    self, parameters: dict[str, Parameter] | None = None
  ) -> None: ...
  def __getitem__(self, key: str) -> Parameter: ...
  def __setitem__(self, key: str, value: Parameter) -> None: ...
  def __contains__(self, key: str) -> bool: ...
  def __len__(self) -> int: ...
  def copy(self) -> Self:
    """Return a deep copy of this ParameterDict."""

  def add(self, param: Parameter) -> None:
    """Add a Parameter, keyed by its ``name``."""

  def update(self, pdict: Self) -> None: ...
  def keys(self) -> list[str]: ...
  def values(self) -> list[Parameter]: ...
  def items(self) -> list[tuple[str, Parameter]]: ...
  @property
  def size(self) -> int:
    """Get the total size of all non-frozen parameters."""

  def as_vector(self, include_frozen: bool = False) -> npt.NDArray[np.float64]:
    """Convert all non-frozen parameters to a flat vector."""

  def as_nominal_vector(
    self, include_frozen: bool = False
  ) -> npt.NDArray[np.float64]:
    """Get the nominal values of parameters as a flat array."""

  def update_from_vector(self, vector: npt.NDArray[np.float64]) -> None:
    """Update all non-frozen parameters from a flat vector."""

  def move_off_bounds(
    self, fraction: float = 0.05, at_bound_tol: float = 1e-3
  ) -> Self:
    """Call ``Parameter.move_off_bound`` on every non-frozen parameter."""

  def save_to_disk(self, path: str | pathlib.Path) -> None:
    """Save the parameter dictionary to disk (schema and data)."""

  @classmethod
  def load_from_disk(cls, path: str | pathlib.Path) -> ParameterDict:
    """Load parameter dictionary from disk (schema and data)."""

  def get_bounds(
    self,
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the bounds for all non-frozen parameters."""

  def reset(self) -> None:
    """Reset all parameters to their nominal values."""

  def sample(
    self, rng: np.random.Generator | None = None
  ) -> npt.NDArray[np.float64]:
    """Sample parameter values within bounds for non-frozen parameters."""

  def randomize(self, rng: np.random.Generator | None = None) -> None:
    """Randomize parameter values for non-frozen parameters."""

  def compare_parameters(
    self,
    init_params: npt.NDArray[np.float64],
    predicted_params: npt.NDArray[np.float64],
    measured_params: npt.NDArray[np.float64] | None = None,
    sig_digits: int = 4,
  ) -> str:
    """Compare true and predicted parameter values."""

  def get_non_frozen_parameter_names(self) -> list[str]: ...
  def get_parameter_info(self) -> str:
    """Get information about all parameters in the dictionary."""
