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

"""Static mjx.warp consumer; ty checks this file but it must never run."""

import ctypes
import functools
from typing import Any

import jax
import numpy as np
import warp as wp
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco import mjx
from mujoco.mjx import warp as mjxw
from mujoco.mjx.third_party.warp._src import jax as wp_jax
from mujoco.mjx.third_party.warp._src.jax import ffi as wp_ffi
from mujoco.mjx.third_party.warp._src.jax import xla_ffi
from mujoco.mjx.warp import (
  bvh,
  collision_driver,
  ffi,
  forward,
  io,
  render,
  render_context,
  smooth,
  testspeed,
  visualize_render,
)
from mujoco.mjx.warp import types as warp_types


def warp_entry_points(m: mjx.Model, d: mjx.Data, mjm: mujoco.MjModel) -> None:
  assert_type(forward.forward(m, d), mjx.Data)
  assert_type(forward.step(m, d), mjx.Data)
  assert_type(collision_driver.collision(m, d), mjx.Data)
  assert_type(smooth.kinematics(m, d), mjx.Data)
  assert_type(smooth.tendon(m, d), mjx.Data)
  assert_type(smooth.com_pos(m, d), mjx.Data)

  ctx = io.create_render_context(
    mjm, 4, devices=['cuda:0', None], cam_res=(64, 64), render_rgb=True
  )
  assert_type(ctx, render_context.RenderContext)
  assert_type(ctx.key, int)
  assert_type(ctx.nworld, int)
  assert_type(ctx.rgb_data_shape, tuple[int, ...])
  assert_type(ctx.seg_data_buffer, wp.array[wp.vec2i])
  handle = ctx.pytree()
  assert_type(handle, render_context.RenderContextPytree)
  assert_type(handle.key, int)
  assert_type(render_context.get(handle), mjxw.mjwp_types.RenderContext)
  assert_type(bvh.refit_bvh(m, d, handle), mjx.Data)
  assert_type(render.render(m, d, handle), list[jax.Array])

  out, batched = forward.step_vmap(2, (m, d), m, d)
  assert_type(out, mjx.Data)
  assert_type(batched, mjx.Data)
  assert_type(
    collision_driver.collision_vmap(2, (m, d), m, d), tuple[mjx.Data, mjx.Data]
  )
  assert_type(
    smooth.kinematics_vmap(2, (m, d), m, d), tuple[mjx.Data, mjx.Data]
  )
  assert_type(
    bvh.refit_bvh_vmap(2, (m, d, handle), m, d, handle),
    tuple[mjx.Data, mjx.Data],
  )
  rendered, rendered_batched = render.render_vmap(
    2, (m, d, handle), m, d, handle
  )
  assert_type(rendered, list[jax.Array])
  assert_type(rendered_batched, list[bool])

  assert_type(mjxw.WARP_INSTALLED, bool)
  assert_type(mjxw.mjwp_io.__name__, str)


def warp_types_module(m: mjx.Model, d: mjx.Data) -> None:
  mw = m._impl
  assert isinstance(mw, warp_types.ModelWarp)
  assert_type(mw.nv_pad, int)
  assert_type(mw.body_tree, tuple[npt.NDArray[np.int32], ...])
  assert_type(mw.M_tiles, tuple[warp_types.TileSet, ...])
  assert_type(mw.block_dim, warp_types.BlockDim)
  assert_type(mw.actuator_armature, jax.Array)
  assert_type(mw.callback, warp_types.Callback)

  dw = d._impl
  assert isinstance(dw, warp_types.DataWarp)
  assert_type(dw.M, jax.Array)
  assert_type(dw.shape, tuple[int, ...])
  assert_type(dw._jax_token, jax.Array)
  assert_type(dw.replace(M=dw.M), warp_types.DataWarp)

  ow = m.opt._impl
  assert isinstance(ow, warp_types.OptionWarp)
  assert_type(ow.graph_mode, wp.JaxCallableGraphMode)
  assert_type(ow.ccd_tolerance, jax.Array)

  tile = warp_types.TileSet(np.zeros(1, dtype=np.int32), 3)
  assert_type(tile.adr, npt.NDArray[np.int32])
  assert_type(tile.size, int)
  assert_type(tile.elemid, npt.NDArray[np.int32])
  assert_type(tile == tile, bool)
  assert_type(hash(tile), int)
  children, aux = tile.tree_flatten()
  assert_type(aux, None)
  assert_type(
    warp_types.TileSet.tree_unflatten(None, children), warp_types.TileSet
  )
  block = warp_types.BlockDim(render=32)
  assert_type(block.render, int)
  assert_type(block.convex_ccd, int)
  assert_type(warp_types.DATA_NON_VMAP, set[str])


def warp_ffi_helpers(m: mjx.Model, d: mjx.Data) -> None:
  def shim(x: wp.array[float], y: wp.array[float]) -> None: ...

  wrapped = ffi.jax_callable_variadic_tuple(
    shim, num_outputs=1, graph_mode=wp.JaxCallableGraphMode.WARP
  )
  assert_type(wrapped(jax.numpy.zeros(3)), list[jax.Array])
  checked = ffi.format_args_for_warp(shim)
  assert_type(checked(wp.zeros(3), wp.zeros(3)), None)

  def entry(m: mjx.Model, d: mjx.Data, flag: int) -> mjx.Data:
    raise NotImplementedError

  marshalled = ffi.marshal_jax_warp_callable(entry)
  assert_type(marshalled(m, d, 3), mjx.Data)

  def vmap_rule(
    size: int, batched: tuple[mjx.Model, mjx.Data], m: mjx.Model, d: mjx.Data
  ) -> tuple[mjx.Data, bool]:
    raise NotImplementedError

  rule = ffi.marshal_custom_vmap(vmap_rule)
  assert_type(rule(2, (m, d), m, d), tuple[mjx.Data, bool])


def benchmarks(mjm: mujoco.MjModel, m: mjx.Model) -> None:
  assert_type(
    testspeed.benchmark(mjm, m, forward.step, nstep=10),
    tuple[float, float, int],
  )
  assert_type(
    testspeed.benchmark_raw_warp(mjm, function='step'),
    tuple[float, float, int],
  )
  assert_type(testspeed.jax_jit, functools.partial[jax.stages.Wrapped])
  assert_type(visualize_render.jax_jit(lambda x: x), jax.stages.Wrapped)
  assert_type(testspeed.main(), None)
  assert_type(visualize_render.main(), None)


def vendored_warp_jax(
  buffer: xla_ffi.XLA_FFI_Buffer,
  frame: xla_ffi.XLA_FFI_CallFrame,
  kernel: wp.Kernel,
) -> None:
  assert_type(wp_jax.get_jax_device(), jax.Device)
  assert_type(wp_jax.device_from_jax(jax.devices()[0]), wp.Device)
  assert_type(wp_jax.device_to_jax('cpu'), jax.Device)
  assert_type(wp_jax.to_jax(wp.zeros(3)), jax.Array)
  assert_type(wp_jax.from_jax(jax.numpy.zeros(3)), wp.array[Any])

  assert_type(wp_ffi.GraphMode, type[wp_ffi.JaxCallableGraphMode])
  assert_type(wp_ffi.JAX_CALLABLE_DEFAULT_GRAPH_CACHE_MAX, int)

  def func(a: wp.array[float], b: wp.array[float]) -> None: ...

  callable_ = wp_ffi.jax_callable(func, graph_mode=wp_ffi.GraphMode.NONE)
  assert_type(callable_, wp_ffi.FfiCallable)
  assert_type(callable_.graph_cache_size, int)
  assert_type(callable_.graph_cache_max, int | None)
  callable_.graph_cache_max = None
  assert_type(callable_.call_descriptors, dict[int, wp_ffi.FfiCallDesc])
  assert_type(callable_(jax.numpy.zeros(3)), list[jax.Array])
  assert_type(callable_.input_args[0].name, str)
  wp_ffi.clear_jax_callable_graph_cache(callable_)

  assert_type(wp_ffi.jax_kernel(kernel), wp_ffi.FfiKernel)
  assert_type(wp_ffi.generate_unique_name(func), str)

  assert_type(buffer.rank, int)
  assert_type(buffer.dims, ctypes._Pointer[ctypes.c_int64])
  assert_type(frame.api.contents.api_version, xla_ffi.XLA_FFI_Api_Version)
  assert_type(xla_ffi.get_device_ordinal_from_callframe(frame), int)
  for key, value in xla_ffi.decode_attrs(frame.attrs).items():
    assert_type(key, str)
    assert not isinstance(value, bytes)
  ctx_ = xla_ffi.ExecutionContext(frame)
  assert_type(ctx_.stage, xla_ffi.XLA_FFI_ExecutionStage)
  assert_type(xla_ffi.FfiBuffer(buffer).shape, tuple[int, ...])
