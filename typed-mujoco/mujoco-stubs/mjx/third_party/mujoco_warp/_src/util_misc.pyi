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

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import types

"""Miscellaneous utilities."""

@wp.func
def is_intersect(
  p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2
) -> bool: ...
@wp.func
def halton(index: int, base: int) -> float: ...
@wp.func
def length_circle(
  p0: wp.vec2, p1: wp.vec2, ind: int, radius: float
) -> float: ...
@wp.func
def wrap_circle(
  end: wp.vec4, side: wp.vec2, radius: float
) -> tuple[float, wp.vec2, wp.vec2]: ...
@wp.func
def wrap_inside(
  end: wp.vec4, radius: float
) -> tuple[float, wp.vec2, wp.vec2]: ...
@wp.func
def wrap(
  x0: wp.vec3,
  x1: wp.vec3,
  pos: wp.vec3,
  mat: wp.mat33,
  radius: float,
  geomtype: int,
  side: wp.vec3,
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def muscle_gain_length(length: float, lmin: float, lmax: float) -> float: ...
@wp.func
def muscle_gain(
  len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: types.vec10
) -> float: ...
@wp.func
def muscle_bias(
  len: float, lengthrange: wp.vec2, acc0: float, prm: types.vec10
) -> float: ...
@wp.func
def muscle_gain_length_deriv(
  length: float, lmin: float, lmax: float
) -> float: ...
@wp.func
def muscle_gain_len_deriv(
  len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: types.vec10
) -> float: ...
@wp.func
def muscle_bias_len_deriv(
  len: float, lengthrange: wp.vec2, acc0: float, prm: types.vec10
) -> float: ...
@wp.func
def muscle_gain_vel_deriv(
  len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: types.vec10
) -> float: ...
@wp.func
def muscle_dynamics_timescale(
  dctrl: float, tau_act: float, tau_deact: float, smooth_width: float
) -> float: ...
@wp.func
def muscle_dynamics(
  control: float, activation: float, prm: types.vec10
) -> float: ...
@wp.func
def dcmotor_slots(dynprm: types.vec10, gainprm: types.vec10) -> types.vec6i: ...
@wp.func
def dcmotor_resistance(
  act_in: wp.array[float],
  worldid: int,
  actadr: int,
  dynprm: types.vec10,
  gainprm: types.vec10,
) -> float: ...
@wp.func
def dcmotor_mech_force(
  act_in: wp.array[float],
  act_dot_in: wp.array[float],
  worldid: int,
  actadr: int,
  length: float,
  dynprm: types.vec10,
  gainprm: types.vec10,
  biasprm: types.vec10,
) -> float: ...
@wp.func
def lugre_stribeck(
  velocity: float, F_C: float, F_S: float, v_S: float
) -> float: ...
@wp.func
def unpack_servo_inputs(
  ctrl_in: wp.array[float], worldid: int, uadr: int, spec: int, u_first: float
) -> wp.vec4: ...
@wp.func
def dcmotor_voltage(
  ctrl_in: wp.array[float],
  worldid: int,
  uadr: int,
  spec: int,
  u_first: float,
  length: float,
  velocity: float,
  x_I: float,
  gainprm: types.vec10,
) -> float: ...
@wp.func
def inside_geom(
  pos: wp.vec3, mat: wp.mat33, size: wp.vec3, geomtype: int, point: wp.vec3
) -> bool: ...
@wp.func
def poly_potential(
  linear: float, poly: wp.vec2, x: float, flg_odd: int
) -> float: ...
@wp.func
def actuator_target_damping(
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array[float],
  actuator_dampingpoly: wp.array[wp.vec2],
  actuator_gear: wp.array[wp.spatial_vectorf],
  worldid: int,
  objtype: int,
  objid: int,
  actuatorid: int,
) -> tuple[float, wp.vec2]: ...
@wp.func
def actuator_target_armature(
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array[float],
  actuator_gear: wp.array[wp.spatial_vectorf],
  worldid: int,
  objtype: int,
  objid: int,
  actuatorid: int,
) -> float: ...
@wp.func
def dof_damping_total(
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_damping: wp.array[float],
  dof_dampingpoly: wp.array[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array[float],
  actuator_dampingpoly: wp.array[wp.vec2],
  actuator_gear: wp.array[wp.spatial_vectorf],
  worldid: int,
  dofid: int,
) -> tuple[float, wp.vec2]: ...
@wp.func
def tendon_damping_total(
  tendon_actuatorid: wp.array[int],
  tendon_damping: wp.array[float],
  tendon_dampingpoly: wp.array[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array[float],
  actuator_dampingpoly: wp.array[wp.vec2],
  actuator_gear: wp.array[wp.spatial_vectorf],
  worldid: int,
  tenid: int,
) -> tuple[float, wp.vec2]: ...
@wp.func
def dof_armature_total(
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_armature: wp.array[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array[float],
  actuator_gear: wp.array[wp.spatial_vectorf],
  worldid: int,
  dofid: int,
) -> float: ...
@wp.func
def tendon_armature_total(
  tendon_actuatorid: wp.array[int],
  tendon_armature: wp.array[float],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_armature: wp.array[float],
  actuator_gear: wp.array[wp.spatial_vectorf],
  worldid: int,
  tenid: int,
) -> float: ...
