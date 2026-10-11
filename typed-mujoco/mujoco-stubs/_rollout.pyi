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

import typing

import numpy
import numpy.typing

__all__: list[str] = ['Rollout']

class Rollout:
  def __init__(
    self, *, nthread: typing.SupportsInt | typing.SupportsIndex
  ) -> None:
    """
    Construct a rollout object containing a thread pool for parallel rollouts.

      input arguments (optional):
        nthread            integer, number of threads in pool
                           if zero, this pool is not started and rollouts run on the calling thread
    """
  def rollout(
    self,
    model: list,
    data: list,
    nstep: typing.SupportsInt | typing.SupportsIndex,
    control_spec: typing.SupportsInt | typing.SupportsIndex,
    state0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    warmstart0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    | None = None,
    control: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    | None = None,
    state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    | None = None,
    sensordata: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    | None = None,
    chunk_size: typing.SupportsInt | typing.SupportsIndex | None = None,
  ) -> None:
    """
    Roll out batch of trajectories from initial states, get resulting states and sensor values.

      input arguments (required):
        model              list of homogenous MjModel instances of length nbatch
        data               list of compatible MjData instances of length nthread
        nstep              integer, number of steps to be taken for each trajectory
        control_spec       specification of controls, ncontrol = mj_stateSize(m, control_spec)
        state0             (nbatch x nstate) nbatch initial state arrays, where
                               nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS)
      input arguments (optional):
        warmstart0         (nbatch x nv)                  nbatch qacc_warmstart arrays
        control            (nbatch x nstep x ncontrol)    nbatch trajectories of nstep controls
      output arguments (optional):
        state              (nbatch x nstep x nstate)      nbatch nstep states
        sensordata         (nbatch x nstep x nsendordata) nbatch trajectories of nstep sensordata arrays
        chunk_size         integer, determines threadpool chunk size. If unspecified, the default is
                               chunk_size = max(1, nbatch / (nthread * 10))
    """
