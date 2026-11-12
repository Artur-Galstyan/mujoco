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
"""Nonlinear Least Squares minimization with box bounds."""

import abc
import dataclasses
import enum
from collections.abc import Callable, Sequence
from typing import Literal, TextIO, TypeAlias

import numpy as np
from numpy import typing as npt

_Matrix: TypeAlias = npt.NDArray[np.float64]
_Real: TypeAlias = npt.NDArray[np.floating | np.integer]

class Verbosity(enum.Enum):
  SILENT = 0
  FINAL = 1
  ITER = 2
  FULLITER = 3

class Status(enum.Enum):
  FACTORIZATION_FAILED = 1
  NO_IMPROVEMENT = 2
  MAX_ITER = 3
  DX_TOL = 4
  G_TOL = 5

@dataclasses.dataclass(frozen=True)
class IterLog:
  """Log of a single iteration of the non-linear least-squares solver."""

  candidate: _Matrix
  objective: float
  reduction: float
  regularizer: float
  residual: _Matrix | None = None
  jacobian: _Matrix | None = None
  grad: _Matrix | None = None
  step: _Matrix | None = None

class Norm(abc.ABC):
  """Abstract interface for norm functions."""

  @abc.abstractmethod
  def value(self, r: _Matrix) -> float:
    """Returns the value of the norm at the input vector `y = norm(r)`."""

  @abc.abstractmethod
  def grad_hess(self, r: _Matrix, proj: _Matrix) -> tuple[_Matrix, _Matrix]:
    """Computes the projected gradient and Hessian of the norm at `r`."""

class Quadratic(Norm):
  """Implementation of the quadratic norm."""

  def value(self, r: _Matrix) -> float:
    """Returns the quadratic norm of `r`."""

  def grad_hess(self, r: _Matrix, proj: _Matrix) -> tuple[_Matrix, _Matrix]:
    """Computes the projected gradient and Hessian of the quadratic norm."""

def least_squares(
  x0: _Real,
  residual: Callable[[_Matrix], _Matrix],
  bounds: Sequence[_Real] | None = ...,
  jacobian: Callable[[_Matrix, _Matrix], _Matrix] | None = ...,
  norm: Norm = ...,
  eps: float = ...,
  mu_min: float = ...,
  mu_max: float = ...,
  mu_factor: float = ...,
  xtol: float = ...,
  gtol: float = ...,
  max_iter: int = ...,
  verbose: Verbosity | int = ...,
  output: TextIO | None = ...,
  iter_callback: Callable[[list[IterLog]], None] | None = ...,
  check_derivatives: bool = ...,
  x_scale: float | _Real | Literal['jac'] | None = ...,
) -> tuple[_Matrix, list[IterLog]]:
  """Nonlinear Least Squares minimization with box bounds."""

def jacobian_fd(
  residual: Callable[[_Matrix], _Matrix],
  x: _Matrix,
  r: _Matrix,
  eps: float,
  n_res: int,
  bounds: list[_Matrix] | None = ...,
) -> tuple[_Matrix, int]:
  """Finite-difference Jacobian of a residual function."""

def check_jacobian(
  residual: Callable[[_Matrix], _Matrix],
  x: _Matrix,
  r: _Matrix,
  jac: _Matrix,
  eps: float,
  n_res: int,
  bounds: list[_Matrix] | None = ...,
  output: TextIO | None = ...,
  name: str | None = ...,
) -> int:
  """Check user-provided Jacobian against internal finite-differencing."""

def check_norm(
  r: _Matrix,
  norm: Norm,
  eps: float,
  output: TextIO | None = ...,
) -> None:
  """Check user-provided norm against internal finite-differencing."""
