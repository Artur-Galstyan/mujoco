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

"""Static MJX types consumer; ty checks this file but it must never run."""

import dataclasses
from typing import Any, Literal

import jax
import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco import mjx
from mujoco.mjx._src import dataclasses as mjx_dataclasses
from mujoco.mjx._src import io as mjx_io
from mujoco.mjx._src import test_util as mjx_test_util
from mujoco.mjx._src import types as mjx_types
from mujoco.mjx._src.support import BindData, BindModel
from mujoco.mjx.warp import test_util as mjxw_test_util

model = mujoco.MjModel.from_xml_string('<mujoco/>')
data = mujoco.MjData(model)

# io: put/make/get round trip.
mx = mjx.put_model(model, impl='jax')
assert_type(mx, mjx.Model)
dx = mjx.make_data(mx)
assert_type(dx, mjx.Data)
dx2 = mjx_io.put_data(model, data, impl=mjx.Impl.JAX)
assert_type(dx2, mjx.Data)
assert_type(mjx.get_data(model, dx), mujoco.MjData | list[mujoco.MjData])
assert_type(mjx_io.get_data_into(data, model, dx), None)
assert_type(mjx_io.has_cuda_gpu_device(), bool)
refs: dict[int, mujoco.MjModel | mujoco.MjData | list[mujoco.MjData]] = {}
mx_cpp = mjx.put_model(model, impl=mjx.Impl.CPP, keepalive_refs=refs)
assert_type(mx_cpp.impl, mjx.Impl)

# io: state helpers.
spec = mujoco.mjtState.mjSTATE_QPOS
assert_type(mjx.state_size(mx, spec), int)
state = mjx.get_state(mx, dx, spec)
assert_type(state, jax.Array)
assert_type(mjx.set_state(mx, dx, state, spec), mjx.Data)
assert_type(mjx.state_size(mx, 3), int)

# Model fields.
assert_type(mx.nq, int)
assert_type(mx.qpos0, jax.Array)
assert_type(mx.opt, mjx.Option)
assert_type(mx.stat, mjx.Statistic)
assert_type(mx.body_parentid, npt.NDArray[np.int32])
assert_type(mx.geom_size, jax.Array)
assert_type(mx.jnt_limited, npt.NDArray[np.bool_])
assert_type(mx.names, bytes)
assert_type(mx.signature, int)
assert_type(mx.opt.timestep, jax.Array)
assert_type(mx.opt.integrator, mjx.IntegratorType)
assert_type(mx.opt.disableflags, mjx.DisableBit)
assert_type(mx.stat.meaninertia, jax.Array)
assert_type(mx.geom_type, npt.NDArray[np.int32])
assert_type(mx.mesh_vert, npt.NDArray[np.float32])
assert_type(mx.tex_data, npt.NDArray[np.uint8])
assert_type(mx.key_qpos, npt.NDArray[np.float64])
assert_type(mx.geom_type, npt.NDArray[np.int32])
assert_type(mx.mesh_vert, npt.NDArray[np.float32])
assert_type(mx.tex_data, npt.NDArray[np.uint8])
assert_type(mx.key_qpos, npt.NDArray[np.float64])

# Data fields.
assert_type(dx.qpos, jax.Array)
assert_type(dx.ctrl, jax.Array)
assert_type(dx.time, jax.Array)
assert_type(dx.eq_active, jax.Array)
assert_type(dx.impl, mjx.Impl)
assert_type(dx[0], mjx.Data)
assert_type(dx.where(jax.numpy.asarray(True), dx2), mjx.Data)

# PyTreeNode: replace and tree_replace keep the type.
dx3 = dx.replace(qpos=dx.qpos + 1.0)
assert_type(dx3, mjx.Data)
assert_type(dx.tree_replace({'qpos': dx.qpos}), mjx.Data)
assert_type(mx.replace(nq=3), mjx.Model)
assert_type(mjx.Data.fields(), tuple[dataclasses.Field[Any], ...])
assert_type(mjx_dataclasses.PyTreeNode.fields()[0].name, str)

# Contact (a PyTreeNode with a synthesized constructor).
contact = mjx.Contact(
  dist=dx.qpos,
  pos=dx.qpos,
  frame=dx.qpos,
  includemargin=dx.qpos,
  friction=dx.qpos,
  solref=dx.qpos,
  solreffriction=dx.qpos,
  solimp=dx.qpos,
  dim=np.zeros(1, np.int32),
  geom1=dx.qpos,
  geom2=dx.qpos,
  geom=dx.qpos,
  efc_address=np.zeros(1, np.int64),
)
assert_type(contact.dist, jax.Array)
assert_type(contact.dim, npt.NDArray[np.int32])
assert_type(contact.efc_address, npt.NDArray[np.int64])


class Pair(mjx_dataclasses.PyTreeNode):
  x: jax.Array
  n: int


pair = Pair(x=dx.qpos, n=2)
assert_type(pair.x, jax.Array)
assert_type(pair.replace(n=3), Pair)

# Enums.
assert_type(mjx.JointType.FREE.dof_width(), int)
assert_type(mjx.JointType.HINGE.qpos_width(), int)
assert_type(mjx.Impl.JAX.value, Literal['jax'])
assert_type(mjx.GeomType.BOX, Literal[mjx.GeomType.BOX])
assert_type(mx.opt.disableflags | mjx.DisableBit.GRAVITY, mjx.DisableBit)
assert_type(mjx.EnableBit.SLEEP, Literal[mjx.EnableBit.SLEEP])
assert_type(mjx.SolverType.NEWTON.value, Literal[2])
assert_type(mjx.SensorType.JOINTPOS, Literal[mjx.SensorType.JOINTPOS])
assert_type(mjx_types.ObjType.BODY, mujoco.mjtObj)

# Pytree path helper.
assert_type(mjx.tree_path_to_attr_str(()), str)


# bind: attached by mjx._src.support at import time.
def _bind_checks(
  m: mjx.Model, d: mjx.Data, s: mujoco.MjStruct, ss: list[mujoco.MjStruct]
) -> None:
  assert_type(m.bind(s), BindModel)
  assert_type(m.bind(ss), BindModel)
  assert_type(d.bind(m, s), BindData)
  assert_type(d.bind(m, ss), BindData)
  assert_type(m.bind(s).model, mjx.Model)
  assert_type(d.bind(m, s).set('x', d.qpos), mjx.Data)


# test_util helpers.
assert_type(mjx.benchmark(model, nstep=2), tuple[float, float, int])
assert_type(mjx_test_util.create_mjcf(0), str)
assert_type(mjx_test_util.p(50), bool)
assert_type(mjx_test_util.load_test_file('x.xml'), mujoco.MjModel)
assert_type(mjx_test_util.efc_order(model, data, dx), np.ndarray)
assert_type(mjxw_test_util.make_data(model, 0), mjx.Data)
assert_type(mjxw_test_util.assert_eq(1.0, 2.0, 'x'), None)
assert_type(mjxw_test_util.load_test_file('x.xml'), mujoco.MjModel)
