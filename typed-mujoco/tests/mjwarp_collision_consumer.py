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

"""Static MJWarp collision consumer; ty checks this file but it must never run."""

import numpy as np
import warp as wp
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco.mjx.third_party.mujoco_warp._src import (
  bvh,
  collision_convex,
  collision_core,
  collision_driver,
  collision_flex,
  collision_gjk,
  collision_primitive,
  collision_primitive_core,
  collision_sdf,
  ray,
  render,
  render_util,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  CollisionType,
  Data,
  GeomType,
  Model,
  RenderContext,
  mat43,
)


def collision_pipeline(m: Model, d: Data) -> None:
  ctx = collision_core.create_collision_context(d.naconmax)
  assert_type(ctx, collision_core.CollisionContext)
  assert_type(ctx.collision_pair, wp.array[wp.vec2i])
  assert_type(ctx.collision_pairid, wp.array[wp.vec2i])
  assert_type(ctx.collision_worldid, wp.array[int])
  assert_type(collision_driver.sap_broadphase(m, d, ctx), None)
  assert_type(collision_driver.nxn_broadphase(m, d, ctx, d.body_awake), None)
  assert_type(collision_driver.collision(m, d), None)
  assert_type(collision_driver.collision(m, d, d.body_awake), None)
  assert_type(
    collision_driver.MJ_COLLISION_TABLE,
    dict[tuple[GeomType, GeomType], CollisionType],
  )
  table = list(collision_driver.MJ_COLLISION_TABLE)
  assert_type(collision_primitive.primitive_narrowphase(m, d, ctx, table), None)
  assert_type(collision_convex.convex_narrowphase(m, d, ctx, table), None)
  assert_type(collision_sdf.sdf_narrowphase(m, d, ctx), None)
  assert_type(collision_flex.flex_collision(m, d, ctx, enable_sat=True), None)
  assert_type(collision_flex.flex_broadphase_aabb(m, d), None)
  assert_type(collision_flex.ENABLE_SAT_PREFILTER, bool)


def kernels_and_builders() -> None:
  kernel = collision_convex.ccd_kernel_builder(0, 1, 12, 12, False, 0, 1, 0)
  assert_type(kernel, wp.Kernel)
  assert_type(
    collision_convex.ccd_hfield_kernel_builder(0, 1, 12, 12, 0, 0), wp.Kernel
  )
  assert_type(bvh.accumulate_flex_vertex_normals(deterministic=True), wp.Kernel)
  assert_type(bvh.compute_bvh_group_roots, wp.Kernel)
  assert_type(render_util.extract_depth_kernel, wp.Kernel)
  assert_type(collision_core.sap_range, wp.Kernel)
  assert_type(render_util.unpack_rgb_kernel, wp.Kernel)
  assert_type(
    collision_convex.vec_maxconpair(), collision_convex.vec_maxconpair
  )
  assert_type(
    collision_primitive_core.vec8f(0.0, 1.0), collision_primitive_core.vec8f
  )
  assert_type(
    collision_primitive_core.mat23f(wp.vec3(), wp.vec3()),
    collision_primitive_core.mat23f,
  )


def analytic_functions(
  g1: collision_core.Geom, g2: collision_core.Geom
) -> None:
  v = wp.vec3()
  assert_type(
    collision_primitive_core.plane_sphere(v, v, v, 1.0), tuple[float, wp.vec3]
  )
  assert_type(
    collision_primitive_core.sphere_sphere(v, 1.0, v, 1.0),
    tuple[float, wp.vec3, wp.vec3],
  )
  assert_type(
    collision_primitive_core.capsule_capsule(v, v, 1.0, 1.0, v, v, 1.0, 1.0),
    tuple[
      wp.vec2, collision_primitive_core.mat23f, collision_primitive_core.mat23f
    ],
  )
  assert_type(g1.pos, wp.vec3)
  assert_type(g1.vert, wp.array[wp.vec3])
  assert_type(g2.margin, float)
  result = collision_gjk.gjk(1e-6, 12, g1, g2, v, v, 0, 1, 0.0, False)
  assert_type(result, collision_gjk.GJKResult)
  assert_type(result.separated, bool)
  assert_type(result.simplex, mat43)
  assert_type(collision_gjk.support(g1, 0, v), collision_gjk.SupportPoint)
  assert_type(collision_gjk.MINVAL, float)


def sdf_functions(volume: collision_sdf.VolumeData) -> None:
  v = wp.vec3()
  assert_type(collision_sdf.sample_volume_sdf(v, volume), float)
  assert_type(collision_sdf.sample_volume_grad(v, volume), wp.vec3)
  assert_type(volume.oct_aabb, wp.array[wp.vec3])
  assert_type(volume.valid, bool)


def bvh_builders(
  mjm: mujoco.MjModel, mjd: mujoco.MjData, rc: RenderContext
) -> None:
  mesh, half = bvh.build_mesh_bvh(mjm, 0, constructor='sah', leaf_size=2)
  assert_type(mesh, wp.Mesh)
  assert_type(half, wp.vec3)
  assert_type(bvh.build_hfield_bvh(mjm, 0), tuple[wp.Mesh, wp.vec3])
  assert_type(bvh.build_flex_bvh(mjm, mjd, 2, 0), tuple[wp.Mesh, wp.array[int]])
  floats: npt.NDArray[np.float32] = np.zeros((4, 3), np.float32)
  adr: npt.NDArray[np.int32] = np.zeros(2, np.int32)
  splats = bvh.build_splat_bvh(floats, floats, floats, floats, adr, adr)
  assert_type(splats[4], wp.Bvh)
  assert_type(splats[-1], int)
  assert_type(bvh.refit_splat_bvh(rc), None)


def rays_and_rendering(
  m: Model, d: Data, mjm: mujoco.MjModel, pnt: wp.array[wp.vec3]
) -> None:
  dist, geomid, normal = ray.ray(m, d, pnt, pnt)
  assert_type(dist, wp.array[float])
  assert_type(geomid, wp.array[int])
  assert_type(normal, wp.array[wp.vec3])
  assert_type(ray.RAY_TOL_REL, float)
  rc = render_util.create_render_context(mjm, nworld=2, cam_res=(8, 8))
  assert_type(rc, RenderContext)
  assert_type(render.render(m, d, rc), None)
  out: wp.array[wp.vec3] = wp.zeros((2, 1, 64), dtype=wp.vec3)
  assert_type(render_util.get_rgb(rc, 0, out), None)
  assert_type(render_util.texture_pixels(mjm, 0), npt.NDArray[np.uint8])
  assert_type(render_util.create_warp_texture(mjm, 0), wp.Texture2D)
  assert_type(render.MAX_SHININESS, float)


def rays_and_flex(ws: collision_flex.FlexWorkspace) -> None:
  v = wp.vec3()
  assert_type(ray.ray_sphere(v, 1.0, v, v), tuple[float, wp.vec3])
  assert_type(ray.ray_plane(v, wp.mat33(), v, v, v), tuple[float, wp.vec3])
  assert_type(ray.ray_geom(v, wp.mat33(), v, v, v, 0), tuple[float, wp.vec3])
  assert_type(ws.dist, wp.array[float])
  assert_type(ws.epa_vert, wp.array[wp.vec3])
  assert_type(ws.nccd, wp.array[int] | None)
  assert_type(
    render_util.compute_ray(0, 1.0, wp.vec2(), wp.vec4(), 8, 8, 0, 0, 0.1),
    tuple[wp.vec3, wp.vec3],
  )
  assert_type(render_util.pack_rgba_to_uint32(0.0, 0.0, 0.0, 1.0), wp.uint32)
