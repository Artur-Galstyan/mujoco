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
"""Run benchmarks for MJX-Warp."""

import functools
from collections.abc import Callable

import jax

import mujoco
from mujoco import mjx

jax_jit: functools.partial[jax.stages.Wrapped]

def benchmark(
  m: mujoco.MjModel,
  mx: mjx.Model,
  step_fn: Callable[[mjx.Model, mjx.Data], mjx.Data],
  nstep: int = ...,
  nenv: int = ...,
  unroll_steps: int = ...,
  render: bool = ...,
) -> tuple[float, float, int]:
  """Benchmark a model."""

def benchmark_raw_warp(
  m: mujoco.MjModel,
  nstep: int = ...,
  nenv: int = ...,
  unroll_steps: int = ...,
  function: str = ...,
  render: bool = ...,
) -> tuple[float, float, int]:
  """Benchmarks raw warp."""

def main() -> None: ...
