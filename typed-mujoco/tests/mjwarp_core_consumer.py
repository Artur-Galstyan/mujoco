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

"""Static MJWarp core consumer; ty checks this file but it must never run."""

import enum
from collections.abc import Callable

import numpy as np
import warp as wp
from etils import epath
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco.mjx.third_party import mujoco_warp as mjw
from mujoco.mjx.third_party.mujoco_warp import viewer
from mujoco.mjx.third_party.mujoco_warp._src import (
  block_cholesky,
  cli,
  history,
  io,
  island,
  set_const,
  sleep,
  types,
  util_misc,
  util_pkg,
  warp_util,
)
from mujoco.mjx.third_party.mujoco_warp._src import math as mjw_math


def check_types() -> None:
  assert_type(mjw.__version__, str)
  mjm = mujoco.MjModel.from_xml_string('<mujoco/>')
  mjd = mujoco.MjData(mjm)

  m = io.put_model(mjm)
  assert_type(m, mjw.Model)
  assert_type(m, types.Model)
  assert_type(io.put_model(mjm, batch_sizes={'geom_size': 2}), types.Model)
  d = io.make_data(mjm, nworld=4, nconmax=100)
  assert_type(d, mjw.Data)
  assert_type(io.put_data(mjm, mjd, nworld=2, njmax=50), types.Data)
  assert_type(mjw.put_model(mjm), mjw.Model)
  assert_type(mjw.make_data(mjm), mjw.Data)

  assert_type(m.nq, int)
  assert_type(m.is_sparse, bool)
  assert_type(m.opt, mjw.Option)
  assert_type(m.opt.timestep, wp.array[float])
  assert_type(m.opt.gravity, wp.array[wp.vec3])
  assert_type(m.opt.broadphase, mjw.BroadphaseType)
  assert_type(m.opt.broadphase_filter, mjw.BroadphaseFilter)
  assert_type(m.geom_size, wp.array[wp.vec3])
  assert_type(m.callback, mjw.Callback)
  assert_type(m.body_tree, tuple[wp.array[int], ...])
  assert_type(m.M_tiles, tuple[types.TileSet, ...])
  assert_type(m.M_tiles[0].adr, wp.array[int])
  assert_type(d.nworld, int)
  assert_type(d.qpos, wp.array[float])
  assert_type(d.xpos, wp.array[wp.vec3])
  assert_type(d.contact, mjw.Contact)
  assert_type(d.contact.dist, wp.array[float])
  assert_type(d.efc, mjw.Constraint)
  assert_type(d.nacon, wp.array[int])
  m.opt.warn_overflow = 0
  m.opt.deterministic = int(mjw.DeterminismType.ALL)
  d.nworld = 3

  def control(model: mjw.Model, data: mjw.Data) -> None:
    pass

  def sensor(model: mjw.Model, data: mjw.Data, stage: types.Stage) -> None:
    pass

  m.callback.control = control
  m.callback.sensor = sensor
  m.callback.passive = None
  sensor_cb: Callable[[mjw.Model, mjw.Data, types.Stage], None] | None = (
    m.callback.sensor
  )
  assert_type(mjw.Callback(control=control).passive is None, bool)
  assert sensor_cb is not None

  io.reset_data(m, d)
  io.reset_data(m, d, reset=wp.array([True, False], dtype=bool))
  io.reset_data_keyframe(m, d, 0)
  io.get_data_into(mjd, mjm, d, world_id=0)
  io.override_model(m, ['opt.iterations=5'])
  io.override_model(mjm, {'opt.iterations': 5, 'opt.cone': 'pyramidal'})
  assert_type(io.is_sparse(mjm), bool)
  layout = io.m_block_layout(mjm)
  assert_type(layout['total'], int)
  assert_type(layout['dof_adr'], npt.NDArray[np.int32])
  assert_type(layout['scalar_tiles'], dict[int, list[int]])
  assert_type(io.find_keys(mjm, 'home'), list[int])
  assert_type(io.make_trajectory(mjm, [0, 1]), npt.NDArray[np.float64])
  assert_type(
    io.load_trajectory('traj.npz', mjm, mjd),
    npt.NDArray[np.float32 | np.float64],
  )
  assert_type(io.CONTACT_EXCLUDE_PASSIVE, int)


def check_enums() -> None:
  solver = mjw.SolverType(2)
  assert_type(solver, mjw.SolverType)
  assert_type(solver.value, int)
  assert_type(mjw.GeomType(6), mjw.GeomType)
  assert_type(mjw.JointType(3), mjw.JointType)
  flags = mjw.DisableBit(0)
  assert_type(flags, mjw.DisableBit)
  assert_type(int(flags | mjw.DisableBit.SPRING), int)
  assert_type(mjw.EnableBit(0), mjw.EnableBit)
  assert_type(mjw.OverflowType(0), mjw.OverflowType)
  assert_type(mjw.SetConstGroup(1), mjw.SetConstGroup)
  assert_type(types.SleepState(0), types.SleepState)
  assert_type(int(mjw.ConeType.PYRAMIDAL), int)
  assert_type(types.Stage(0), types.Stage)
  assert_type(viewer.EngineOptions(0), viewer.EngineOptions)
  assert_type(int(viewer.EngineOptions.C), int)
  assert issubclass(mjw.SolverType, enum.IntEnum)
  assert_type(types.MJ_MINVAL, float)
  assert_type(types.ROD_HINGE_SIZE, int)
  assert_type(types.BlockDim().render, int)


def check_vec_types() -> None:
  v = types.vec5f(1.0, 2.0, 3.0, 4.0, 5.0)
  assert_type(v, types.vec5f)
  assert_type(v[0], float)
  v[0] = 2.0
  assert_type(len(v), int)
  assert_type(list(v), list[float])
  assert_type(types.vec6i()[0], int)
  assert_type(types.vec128().__len__(), int)
  mat = types.mat66f()
  assert_type(mat[0, 1], float)
  assert_type(mat.get_col(0)[1], float)
  mat[0, 1] = 1.0
  mat.set_row(0, mat.get_row(1))
  assert_type(len(mat), int)
  assert_type(types.vec10(), types.vec10f)
  assert_type(types.array('nq', float), wp.array)


def check_warp_util() -> None:
  with warp_util.EventTracer() as tracer:
    trace = tracer.trace()
    assert_type(tracer, warp_util.EventTracer)
    for name, (events, sub) in trace.items():
      assert_type(name, str)
      assert_type(events, tuple[float, ...])
      assert_type(
        sub, dict[str, tuple[tuple[float, ...], warp_util._EventTrace]]
      )

  def step(m: mjw.Model, d: mjw.Data) -> int:
    return d.nworld

  scoped: Callable[[mjw.Model, mjw.Data], int] = warp_util.event_scope(step)
  cached: Callable[[mjw.Model, mjw.Data], int] = warp_util.cache_kernel(step)
  assert scoped is not cached
  assert_type(warp_util.is_array_spec(wp.array[float]), bool)
  warp_util.check_toolkit_driver()


def check_math() -> None:
  q = wp.quat(0.0, 0.0, 0.0, 1.0)
  p = wp.vec3(1.0, 2.0, 3.0)
  assert_type(mjw_math.mul_quat(q, q), wp.quat)
  assert_type(mjw_math.quat_to_mat(q), wp.mat33)
  assert_type(mjw_math.rot_vec_quat(p, q), wp.vec3)
  assert_type(mjw_math.orthogonals(p), tuple[wp.vec3, wp.vec3])
  assert_type(mjw_math.make_frame(p), wp.mat33)
  assert_type(mjw_math.gjk_normalize(p), tuple[wp.vec3, bool])
  assert_type(mjw_math.normalize_with_norm(p), tuple[wp.vec3, float])
  assert_type(mjw_math.safe_div(1.0, 2.0), float)
  assert_type(mjw_math.upper_tri_index(3, 0, 1), int)
  assert_type(mjw_math.eig3(wp.mat33()), tuple[wp.vec3, wp.mat33])
  assert_type(
    mjw_math.lu_factor_6x6(types.mat66f()),
    tuple[types.mat66f, types.vec6i, bool],
  )
  assert_type(util_misc.halton(3, 2), float)
  assert_type(util_misc.muscle_gain_length(1.0, 0.5, 1.5), float)
  assert_type(
    util_misc.muscle_gain(1.0, 0.0, wp.vec2(0.0, 1.0), 1.0, types.vec10f()),
    float,
  )
  assert_type(
    util_misc.is_intersect(wp.vec2(), wp.vec2(), wp.vec2(), wp.vec2()), bool
  )
  assert_type(util_pkg.check_version('mujoco>=3.0'), bool)
  assert_type(block_cholesky.ctz64(wp.uint64(4)), int)
  assert_type(
    block_cholesky.create_blocked_cholesky_solve_func(16, 32), wp.Function
  )


def check_pipeline(m: mjw.Model, d: mjw.Data) -> None:
  set_const.set_const(m, d, groups=mjw.SetConstGroup.ALL, restore=False)
  set_const.set_const_fixed(m, d)
  set_const.set_length_range(m, d, index=-1)
  sleep.update_sleep(m, d, flg_staticawake=1)
  sleep.wake(m, d)
  sleep.sleep(m, d)
  assert_type(sleep.K_AWAKE_VAL, int)
  island.island(m, d)
  island.direct_dsu(m, d, wp.zeros(4, dtype=int))
  island.compute_island_mapping(m, d)
  history.reset_history(m, d)
  history.read_ctrl_delayed(m, d, d.ctrl)
  history.insert_ctrl_history(m, d)
  history.read_ctrl(m, d, 0, d.time, -1, d.ctrl)
  history.init_sensor_history(m, d, 0, None, d.sensordata, phase=0.5)


def check_cli() -> None:
  assert_type(cli.NWORLD.value, int)
  assert_type(cli.NCONMAX.value, int | None)
  assert_type(cli.OVERRIDE.value, list[str])
  assert_type(cli.EVENT_TRACE.value, bool)
  assert_type(cli.DEVICE.value, str | None)
  mjm = cli.load_model(epath.Path('model.xml'))
  assert_type(mjm, mujoco.MjModel)
  m, d, rc, ctrls = cli.init_structs(mjw.step, mjm)
  assert_type(m, mjw.Model)
  assert_type(rc, mjw.RenderContext | None)
  assert_type(cli.unroll(mjw.step, m, d, rc, ctrls=ctrls), float)
  viewer.key_callback(32)
  viewer.main()
