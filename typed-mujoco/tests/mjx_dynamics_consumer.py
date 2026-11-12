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

"""Static MJX dynamics consumer; ty checks this file but it must never run."""

import jax
import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco.mjx._src import (
  constraint,
  derivative,
  forward,
  inverse,
  passive,
  scan,
  sensor,
  smooth,
  solver,
  support,
)
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src.types import Data, Model

_Int = npt.NDArray[np.int_]


def math_functions(a: jax.Array, b: jax.Array, axis: tuple[int, ...]) -> None:
  assert_type(mjx_math.safe_div(1.0, 2.0), float)
  assert_type(mjx_math.safe_div(a, 2.0), jax.Array)
  assert_type(mjx_math.safe_div(2.0, a), jax.Array)
  assert_type(mjx_math.norm(a, axis), jax.Array)
  assert_type(mjx_math.normalize_with_norm(a), tuple[jax.Array, jax.Array])
  assert_type(mjx_math.quat_mul(a, b), jax.Array)
  assert_type(mjx_math.quat_to_axis_angle(a), tuple[jax.Array, jax.Array])
  assert_type(mjx_math.transform_motion(a, b, a), jax.Array)
  assert_type(mjx_math.motion_cross(a, b), jax.Array)
  assert_type(mjx_math.orthogonals(a), tuple[jax.Array, jax.Array])


def scan_functions(m: Model, x: jax.Array) -> None:
  def per_joint(q: jax.Array) -> jax.Array:
    return q

  assert_type(scan.flat(m, per_joint, 'j', 'j', x, group_by='j'), jax.Array)
  assert_type(scan.flat(m, per_joint, 'u', 'u', x, group_by='u'), jax.Array)

  def carry(
    y: tuple[jax.Array, jax.Array], q: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    return y

  assert_type(
    scan.body_tree(m, carry, 'b', 'b', x, reverse=True),
    tuple[jax.Array, jax.Array],
  )


def forward_functions(m: Model, d: Data) -> None:
  assert_type(forward.step(m, d), Data)
  assert_type(forward.forward(m, d), Data)
  assert_type(forward.step(m=m, d=d), Data)
  assert_type(forward.named_scope(smooth.crb, 'crb')(m, d), Data)
  assert_type(forward.fwd_velocity(m, d), Data)
  assert_type(forward.fwd_actuation(m, d), Data)
  assert_type(forward.fwd_acceleration(m, d), Data)
  assert_type(forward.fwd_position(m, d), Data)
  assert_type(forward.euler(m, d), Data)
  assert_type(forward.implicit(m, d), Data)
  assert_type(forward.rungekutta4(m, d), Data)


def smooth_functions(m: Model, d: Data, x: jax.Array) -> None:
  assert_type(smooth.kinematics(m, d), Data)
  assert_type(smooth.crb(m, d), Data)
  assert_type(smooth.rne(m, d, flg_acc=True), Data)
  assert_type(smooth.solve_m(m, d, x), jax.Array)
  assert_type(smooth.tendon_dot(m, d), jax.Array)


def solver_functions(m: Model, d: Data) -> None:
  assert_type(solver.solve(m, d), Data)
  ctx = solver.Context.create(m, d, grad=False)
  assert_type(ctx, solver.Context)
  assert_type(ctx.qacc, jax.Array)
  assert_type(ctx.cost, jax.Array)
  assert_type(ctx.replace(cost=ctx.prev_cost), solver.Context)


def constraint_functions(m: Model, d: Data, mj: mujoco.MjModel) -> None:
  assert_type(constraint.make_constraint(m, d), Data)
  efc_type = constraint.make_efc_type(mj)
  assert_type(efc_type, _Int)
  assert_type(constraint.make_efc_address(m, efc_type, efc_type), _Int)
  assert_type(
    constraint.counts(efc_type),
    tuple[np.int64, np.int64, np.int64, np.int64],
  )


def other_dynamics(m: Model, d: Data) -> None:
  assert_type(passive.passive(m, d), Data)
  assert_type(derivative.deriv_smooth_vel(m, d), jax.Array | None)
  assert_type(inverse.inverse(m, d), Data)
  assert_type(inverse.discrete_acc(m, d), Data)
  assert_type(inverse.inv_constraint(m, d), Data)
  assert_type(sensor.sensor_pos(m, d), Data)
  assert_type(sensor.sensor_vel(m, d), Data)
  assert_type(sensor.sensor_acc(m, d), Data)


def support_functions(
  m: Model,
  d: Data,
  mj: mujoco.MjModel,
  spec: mujoco.MjsJoint,
  v: jax.Array,
  p: jax.Array,
) -> None:
  assert_type(support.is_sparse(m), bool)
  assert_type(support.full_m(m, d), jax.Array)
  assert_type(support.mul_m(m, d, v), jax.Array)
  assert_type(support.jac(m, d, p, v), tuple[jax.Array, jax.Array])
  assert_type(support.apply_ft(m, d, v, v, p, v), jax.Array)
  assert_type(support.contact_force(m, d, 0), jax.Array)
  assert_type(
    support.contact_force_dim(m, d, 3),
    tuple[jax.Array, npt.NDArray[np.intp]],
  )
  assert_type(support.id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, 0), str | None)
  assert_type(support.name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'j'), int)
  bound = support.BindModel(m, [spec])
  assert_type(bound.prefix, str)
  assert_type(bound.id, int | list[int])
  assert_type(bound.range, jax.Array)
  bound_data = support.BindData(d, m, [spec])
  assert_type(bound_data.qpos, jax.Array)
  assert_type(bound_data.id, int | list[int])
  assert_type(bound_data.prefix, str)
  assert_type(bound_data.model, Model)
  assert_type(bound_data.data, Data)
  assert_type(bound_data.set('qpos', v), Data)
  assert_type(support.local_to_global(p, v, p, v), tuple[jax.Array, jax.Array])
  assert_type(support.muscle_gain_length(v, v, v), jax.Array)
  assert_type(support.wrap_circle(v, v, v, v), tuple[jax.Array, jax.Array])


def more_checks(m: Model, d: Data, v: jax.Array) -> None:
  assert_type(smooth.com_pos(m, d), Data)
  assert_type(smooth.transmission(m, d), Data)
  assert_type(smooth.tendon_bias(m, d), Data)
  assert_type(support.xfrc_accumulate(m, d), jax.Array)
  assert_type(support.make_m(m, v, v), jax.Array)
  assert_type(support.muscle_dynamics(v, v, v), jax.Array)
  assert_type(mjx_math.quat_integrate(v, v, v), jax.Array)
  assert_type(
    mjx_math.closest_segment_point_and_dist(v, v, v),
    tuple[jax.Array, jax.Array],
  )
