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

"""Run open-loop MuJoCo simulations without a viewer.

Use `rollout` for one call or `Rollout` to reuse a thread pool.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Literal, TypeAlias, overload

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

from mujoco import MjData, MjModel

_NumArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_Model: TypeAlias = MjModel | Sequence[MjModel]
_Data: TypeAlias = MjData | Sequence[MjData]

class Rollout:
  """Reuse a thread pool for batches of open-loop trajectories.

  Use a context manager to release the pool, or call `close` explicitly.

  Example (no GUI required):
    >>> import mujoco
    >>> import numpy as np
    >>> from mujoco import rollout
    >>> model = mujoco.MjModel.from_xml_string(
    ...   '<mujoco><worldbody><body><joint name="j" type="slide"/>'
    ...   '<geom size=".1"/></body></worldbody><actuator>'
    ...   '<motor joint="j"/></actuator><sensor>'
    ...   '<jointpos joint="j"/></sensor></mujoco>'
    ... )
    >>> data = mujoco.MjData(model)
    >>> nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    >>> initial = np.zeros(nstate, dtype=mujoco.MJTNUM_DTYPE)
    >>> with rollout.Rollout(nthread=0) as pool:
    ...   states, sensors = pool.rollout(model, data, initial, nstep=2)
    >>> states.shape == (1, 2, nstate)
    True
    >>> sensors.shape == (1, 2, model.nsensordata)
    True
  """

  nthread: int

  def __init__(self, *, nthread: int | None = None) -> None:
    """Create a pool with `nthread` workers; zero or None uses this thread."""
  def __enter__(self) -> Self: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None: ...
  def close(self) -> None:
    """Shut down the pool; later calls to `rollout` raise RuntimeError."""
  @overload
  def rollout(
    self,
    model: _Model,
    data: _Data,
    initial_state: npt.ArrayLike,
    control: npt.ArrayLike | None = None,
    *,
    control_spec: int = ...,
    skip_checks: Literal[False] = False,
    nstep: int | None = None,
    initial_warmstart: npt.ArrayLike | None = None,
    state: npt.ArrayLike | None = None,
    sensordata: npt.ArrayLike | None = None,
    chunk_size: int | None = None,
  ) -> tuple[_NumArray, _NumArray]:
    """Simulate `nbatch` trajectories for `nstep` steps.

    `initial_state` holds full-physics states of size
    `mj_stateSize(model, mjSTATE_FULLPHYSICS)`, with shape `(nbatch, nstate)`.
    `control` has shape `(nbatch, nstep, ncontrol)`, where `ncontrol` is
    `mj_stateSize(model, control_spec)` (CTRL by default). A single batch or
    step can be broadcast; omit `control` to apply no new control vectors.
    `initial_warmstart` has shape `(nbatch, model.nv)` when supplied.
    `model` may be one model or `nbatch` compatible models; supply a list of
    one `data` per worker (or a single `data` without workers).

    Infers batch and step counts from inputs, `nstep`, or output shapes.
    Returns post-step `(state, sensordata)` with shapes
    `(nbatch, nstep, nstate)` and
    `(nbatch, nstep, model.nsensordata)`. Allocates missing outputs using
    `mujoco.MJTNUM_DTYPE`; provided arrays must have those shapes.
    """
  @overload
  def rollout(
    self,
    model: _Model,
    data: _Data,
    initial_state: npt.ArrayLike,
    control: npt.ArrayLike | None = None,
    *,
    control_spec: int = ...,
    skip_checks: bool,
    nstep: int | None = None,
    initial_warmstart: npt.ArrayLike | None = None,
    state: npt.ArrayLike | None = None,
    sensordata: npt.ArrayLike | None = None,
    chunk_size: int | None = None,
  ) -> tuple[npt.ArrayLike | None, npt.ArrayLike | None]:
    """With `skip_checks=True`, call the native rollout without Python setup.

    No batch or step inference, broadcasting, output allocation, or Python
    shape/type checks occur. Pass `model` and `data` as lists, set `nstep`,
    and supply C-contiguous arrays with the exact sizes above and dtype
    `mujoco.MJTNUM_DTYPE`. The native code still checks array sizes.
    Returns the supplied `state` and `sensordata`, even if either is None.
    """

persistent_rollout: Rollout | None

def shutdown_persistent_pool() -> None:
  """Close the optional shared rollout pool; also called at interpreter exit."""

@overload
def rollout(
  model: _Model,
  data: _Data,
  initial_state: npt.ArrayLike,
  control: npt.ArrayLike | None = None,
  *,
  control_spec: int = ...,
  skip_checks: Literal[False] = False,
  nstep: int | None = None,
  initial_warmstart: npt.ArrayLike | None = None,
  state: npt.ArrayLike | None = None,
  sensordata: npt.ArrayLike | None = None,
  chunk_size: int | None = None,
  persistent_pool: bool = False,
) -> tuple[_NumArray, _NumArray]:
  """Run trajectories, closing the temporary pool after the call.

  `persistent_pool=True` reuses a shared pool until
  `shutdown_persistent_pool()` is called. `initial_state` has shape
  `(nbatch, nstate)`, where `nstate = mj_stateSize(model, mjSTATE_FULLPHYSICS)`;
  `control` has shape `(nbatch, nstep, ncontrol)`, where
  `ncontrol = mj_stateSize(model, control_spec)`. Returns post-step state
  `(nbatch, nstep, nstate)` and sensor values
  `(nbatch, nstep, model.nsensordata)`. Infers batch and step counts when
  possible and allocates missing outputs with `mujoco.MJTNUM_DTYPE`.

  Example (no GUI required):
    >>> import mujoco
    >>> import numpy as np
    >>> from mujoco import rollout
    >>> model = mujoco.MjModel.from_xml_string(
    ...   '<mujoco><worldbody><body><joint name="j" type="slide"/>'
    ...   '<geom size=".1"/></body></worldbody><actuator>'
    ...   '<motor joint="j"/></actuator><sensor>'
    ...   '<jointpos joint="j"/></sensor></mujoco>'
    ... )
    >>> data = mujoco.MjData(model)
    >>> nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    >>> initial = np.zeros(nstate, dtype=mujoco.MJTNUM_DTYPE)
    >>> controls = np.zeros((3, model.nu), dtype=mujoco.MJTNUM_DTYPE)
    >>> states, sensors = rollout.rollout(model, data, initial, controls)
    >>> states.shape == (1, 3, nstate)
    True
    >>> sensors.shape
    (1, 3, 1)
  """

@overload
def rollout(
  model: _Model,
  data: _Data,
  initial_state: npt.ArrayLike,
  control: npt.ArrayLike | None = None,
  *,
  control_spec: int = ...,
  skip_checks: bool,
  nstep: int | None = None,
  initial_warmstart: npt.ArrayLike | None = None,
  state: npt.ArrayLike | None = None,
  sensordata: npt.ArrayLike | None = None,
  chunk_size: int | None = None,
  persistent_pool: bool = False,
) -> tuple[npt.ArrayLike | None, npt.ArrayLike | None]:
  """Skip Python checks, inference, broadcasting, and output allocation.

  With `skip_checks=True`, pass `model` as a list and set `nstep`. Supply
  correctly sized C-contiguous arrays of dtype `mujoco.MJTNUM_DTYPE`; native
  size checks still apply. Returns the supplied outputs, even if either is
  None. Set `persistent_pool=True` to reuse the pool; otherwise it closes.
  """
