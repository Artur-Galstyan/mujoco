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

"""Static MJX collision consumer; ty checks this file but it must never run."""

from collections.abc import Callable, Iterator, Sequence
from typing import NoReturn

import jax
import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco.mjx import testspeed, viewer
from mujoco.mjx._src import (
  bvh,
  collision_convex,
  collision_driver,
  collision_primitive,
  collision_sdf,
  collision_types,
  mesh,
  ray,
  render,
  render_util,
)
from mujoco.mjx._src.types import Data, GeomType, Impl, Model
from mujoco.mjx.codegen import (
  file,
  generate_warp_shim,
  generate_warp_types,
  trace,
)
from mujoco.mjx.warp.render_context import RenderContextPytree


def collision_types_api(
  pos: jax.Array, mat: jax.Array, size: jax.Array
) -> None:
  info = collision_types.GeomInfo(pos, mat, size)
  assert_type(info.pos, jax.Array)
  assert_type(info.replace(pos=pos), collision_types.GeomInfo)
  key = collision_types.FunctionKey((1, 2), (-1, -1), 3)
  assert_type(key.types, tuple[int, int])
  assert_type(key.subgrid_size, tuple[int, int])
  hfield = collision_types.HFieldInfo(pos, mat, np.zeros(4), 2, 2, size)
  assert_type(hfield.size, npt.NDArray[np.float64])
  assert_type(hfield.nrow, int)
  result: collision_types.Collision = (pos, pos, mat)
  assert_type(result[0], jax.Array)


def collider_api(
  m: Model, d: Data, key: collision_types.FunctionKey, geom: jax.Array
) -> None:
  assert_type(collision_primitive.plane_sphere.ncon, int)
  assert_type(
    collision_primitive.plane_sphere(m, d, key, geom),
    tuple[jax.Array, jax.Array, jax.Array],
  )
  assert_type(collision_sdf.cylinder_cylinder.ncon, int)
  contact = collision_sdf.sphere_cylinder(m, d, key, geom)
  assert_type(contact[0], jax.Array)
  assert_type(collision_convex.convex_convex.ncon, int)
  contact = collision_convex.hfield_sphere(m, d, key, geom)
  assert_type(contact[2], jax.Array)

  def sphere_fn(
    a: collision_types.GeomInfo, b: collision_types.GeomInfo
  ) -> collision_types.Collision:
    return a.pos, a.mat, b.size

  wrapped = collision_sdf.collider(ncon=1)(sphere_fn)
  assert_type(wrapped.ncon, int)
  assert_type(wrapped(m, d, key, geom), collision_types.Collision)

  def box_fn(
    a: collision_types.ConvexInfo, b: collision_types.ConvexInfo
  ) -> collision_types.Collision:
    return a.pos, a.mat, b.size

  convex_wrapped = collision_convex.collider(ncon=4)(box_fn)
  assert_type(convex_wrapped.ncon, int)
  assert_type(convex_wrapped(m, d, key, geom), collision_types.Collision)

  def hfield_fn(
    h: collision_types.HFieldInfo,
    g: collision_types.GeomInfo,
    subgrid_size: tuple[int, int],
  ) -> collision_types.Collision:
    return g.pos, g.mat, h.data

  collision_convex.collider(ncon=1)(hfield_fn)
  state = collision_sdf.GradientState(geom, geom)
  assert_type(state.dist, jax.Array)

  def identity(p: jax.Array) -> jax.Array:
    return p

  sdf: collision_sdf.SDFFn = identity
  assert_type(sdf(geom), jax.Array)


def driver_api(m: Model, mj_model: mujoco.MjModel, d: Data) -> None:
  assert_type(collision_driver.collision(m, d), Data)
  assert_type(
    collision_driver.has_collision_fn(GeomType.PLANE, GeomType.SPHERE), bool
  )
  assert_type(
    collision_driver.geom_pairs(mj_model), Iterator[tuple[int, int, int]]
  )
  assert_type(collision_driver.make_condim(m, Impl.JAX), npt.NDArray[np.int32])
  assert_type(collision_driver.make_condim(mj_model), npt.NDArray[np.int32])


def mesh_api(
  m: Model, mj_model: mujoco.MjModel, info: collision_types.GeomInfo
) -> None:
  assert_type(mesh.box(info), collision_types.ConvexInfo)
  assert_type(mesh.hfield(mj_model, 0), collision_types.HFieldInfo)
  assert_type(mesh.hfield(m, 0).data, jax.Array)
  assert_type(mesh.hfield_prism(info.pos), collision_types.ConvexInfo)
  assert_type(mesh.convex(mj_model, 0).vert, jax.Array)


def ray_api(m: Model, d: Data, pnt: jax.Array, vec: jax.Array) -> None:
  dist, geom_id = ray.ray(m, d, pnt, vec, geomgroup=(1, 0), bodyexclude=[1, 2])
  assert_type(dist, jax.Array)
  assert_type(geom_id, jax.Array)
  assert_type(ray.ray_geom(pnt, pnt, vec, GeomType.SPHERE), jax.Array)


def render_api(
  m: Model, d: Data, ctx: RenderContextPytree, packed: jax.Array
) -> None:
  rgb, depth, d2 = render.render(m, d, ctx)
  assert_type(rgb, jax.Array)
  assert_type(d2, Data)
  seg_out = render.render_with_segmentation(m, d, ctx)
  assert_type(seg_out[2], jax.Array)
  assert_type(seg_out[3], Data)
  assert_type(bvh.refit_bvh(m, d, ctx), Data)
  assert_type(render_util.get_rgb(ctx, 0, rgb), jax.Array)
  assert_type(render_util.get_depth(ctx, 0, depth, 1.0), jax.Array)
  assert_type(render_util.get_segmentation(ctx, 0, packed), jax.Array)


def codegen_api(src: str) -> None:
  assert_type(file.get_base_path().name, str)
  annotations = file.get_cls_type_annotations(src)
  assert_type(annotations, dict[str, dict[str, str]])
  infos = trace.get_mjx_warp_field_info(src, file.get_cls_type_annotations)
  assert_type(infos, dict[str, trace.FieldInfo])
  usage = trace.trace_function('a.py', 'f', infos)
  assert_type(usage, trace.FieldUsage)
  assert_type(usage.model_fields, Sequence[str])
  assert_type(usage.render_context_in_caller, bool)
  assert_type(infos['x'].param_order, tuple[int, str])
  generate_warp_shim.create_jax_warp_shim(
    'f', usage, infos, infos, file.get_base_path()
  )
  generate_warp_types.write_header(file.get_base_path())
  generate_warp_types.write_core_cls(
    'Model', file.get_base_path(), file.get_base_path(), set_diff=False
  )


def entry_points() -> None:
  viewer.key_callback(32)


VIEWER_MAIN: Callable[[], NoReturn] = viewer.main
SPEED_MAIN: Callable[[], NoReturn] = testspeed.main
