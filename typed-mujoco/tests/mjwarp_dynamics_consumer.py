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

"""Static mjwarp dynamics consumer; ty checks this file but it must never run."""

from typing import Literal

import warp as wp
from typing_extensions import assert_type

from mujoco.mjx.third_party.mujoco_warp._src import (
  constraint,
  derivative,
  forward,
  inverse,
  passive,
  sensor,
  smooth,
  solver,
  support,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  InverseContext,
  Model,
  SolverContext,
  mat43,
  mat63,
  vec6,
)


def check_forward(m: Model, d: Data) -> None:
  assert_type(forward.step(m, d), None)
  assert_type(forward.forward(m, d), None)
  assert_type(forward.fwd_position(m, d, factorize=False), None)
  assert_type(forward.fwd_acceleration(m, d, True), None)
  assert_type(forward.implicit_qlu(m, d), None)
  assert_type(forward.implicit_qlu(m, d, d.qLU), None)
  forward.euler(m, d)
  forward.rungekutta4(m, d)
  forward.step1(m, d)
  forward.step2(m, d)


def check_smooth(m: Model, d: Data, qfrc: wp.array[float]) -> None:
  assert_type(smooth.skip_qacc_smooth(m, d), bool)
  assert_type(smooth.rne(m, d, flg_acc=True), None)
  assert_type(smooth.tendon_bias(m, d, qfrc), None)
  assert_type(smooth.solve_m(m, d, qfrc, qfrc), None)
  smooth.solve_m(m, d, qfrc, qfrc, x_scratch=qfrc)
  smooth.solve_LD(m, d, d.qLD, d.qLDiagInv, qfrc, qfrc)
  smooth.factor_solve_i(m, d, d.M, d.qLD, d.qLDiagInv, qfrc, qfrc)
  smooth.kinematics(m, d)
  smooth.crb(m, d)
  smooth.transmission(m, d)
  force = wp.vec3()
  assert_type(smooth.transform_force(force, force, force), wp.spatial_vectorf)


def check_solver(m: Model, d: Data, ctx: SolverContext) -> None:
  inv_ctx = solver.create_inverse_context(m, d)
  assert_type(inv_ctx, InverseContext)
  assert_type(solver.init_context(m, d, ctx), None)
  solver.init_context(m, d, inv_ctx, grad=False, compact=True)
  solver.solve(m, d)
  solver.noslip(m, d)
  assert_type(constraint.make_constraint(m, d), None)
  assert_type(constraint.regularize_constraint(m, d), None)


def check_passive(m: Model, d: Data, vec: wp.array[float]) -> None:
  efm_con = passive.build_efm_contact(m, d)
  assert_type(
    efm_con,
    tuple[
      wp.array[int],
      wp.array[float],
      wp.array[float],
      wp.array[float],
      wp.array[int],
    ],
  )
  scratch = passive.flex_stretch_scratch(m, d)
  assert_type(scratch, tuple[wp.array[wp.vec3], wp.array[wp.vec3]])
  passive.flex_stretch_mul(m, d, vec, vec, scratch=scratch)
  passive.flex_hessian_mul(
    m, d, scratch[0], scratch[1], 1.0, 0.0, skip=wp.zeros(1, dtype=bool)
  )
  passive.passive(m, d)
  assert_type(passive.geom_semiaxes(wp.vec3(), 1), wp.vec3)
  assert_type(passive.ellipsoid_max_moment(wp.vec3(), 0), float)
  assert_type(
    passive.stretch_edge_endpoints(3), wp.Matrix[int, Literal[6], Literal[2]]
  )
  assert_type(passive.stretch_edge_vectors(mat43(), 3), mat63)
  elongation = vec6()
  assert_type(
    passive.stretch_tension(passive.stretch_metric(vec, 0, 6), elongation, 6),
    vec6,
  )


def check_derivative(m: Model, d: Data, vec: wp.array[float]) -> None:
  efm_con, flex_scratch = derivative.eff_mul_m_scratch(m, d)
  derivative.eff_mul_m(m, d, vec, vec, None, efm_con, flex_scratch)
  derivative.eff_prec(m, d, vec, vec, x_scratch=vec, tw_diag=vec)
  derivative.eff_solve(m, d, vec)
  derivative.deriv_smooth_vel(m, d, vec)
  assert_type(derivative.deriv_rne_cvel_cdof_dot, wp.Kernel)
  assert_type(derivative.deriv_rne_cfrcbody_backward(True), wp.Kernel)
  assert_type(derivative.deriv_rne_body2jnt_sparse(), wp.Kernel)
  wp.launch(derivative.deriv_rne_cvel_cdof_dot, dim=1, inputs=[])


def check_inverse_sensor(m: Model, d: Data, qacc: wp.array[float]) -> None:
  assert_type(inverse.inverse(m, d), None)
  inverse.discrete_acc(m, d, qacc)
  inverse.inv_constraint(m, d)
  sensor.sensor_pos(m, d)
  sensor.sensor_vel(m, d)
  sensor.sensor_acc(m, d, skip_rne_postconstraint=True)
  assert_type(sensor.energy_pos(m, d), None)
  sensor.energy_vel(m, d)


def check_support(m: Model, d: Data, vec: wp.array[float]) -> None:
  support.mul_m(m, d, vec, vec)
  support.mul_m(m, d, vec, vec, skip=wp.zeros(1, dtype=bool), M=d.M)
  support.xfrc_accumulate(m, d, vec)
  support.apply_ft(m, d, wp.zeros(1, dtype=wp.spatial_vectorf), vec, True)
  point = wp.zeros(1, dtype=wp.vec3)
  body = wp.zeros(1, dtype=int)
  support.jac(m, d, vec, None, point, body)
  support.get_state(m, d, vec, 1, active=wp.zeros(1, dtype=bool))
  support.set_state(m, d, vec, 1)
  support.contact_force(m, d, body, True, wp.zeros(1, dtype=wp.spatial_vectorf))
  assert_type(support.MAX_INTERP_BODIES, int)
  assert_type(support.mul_m_kernel(True), wp.Kernel)
  assert_type(support.mul_m_dense(6, False), wp.Kernel)
  assert_type(support.contact_force_kernel, wp.Kernel)
  assert_type(support.flex_phi(0.5, 1, 2), float)
  assert_type(support.mat33_to_quat_polar(wp.mat33()), wp.quat)
  frc = wp.spatial_vectorf()
  assert_type(support.transform_force(frc, wp.vec3()), wp.spatial_vectorf)
  assert_type(
    support.transform_force(wp.vec3(), wp.vec3(), wp.vec3()),
    wp.spatial_vectorf,
  )
  weights = support.vec_interp_weight()
  assert_type(weights, support.vec_interp_weight)
