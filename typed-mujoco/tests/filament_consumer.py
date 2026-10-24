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
"""Type-check the optional Filament API; do not execute this consumer."""

import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import model_scene, renderer


def check_filament_types(
  ctx: mjrf.Context,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  scene: mjrf.Scene,
  target: mjrf.RenderTarget,
  light: mjrf.Light,
  renderable: mjrf.Renderable,
  option: mujoco.MjvOption,
  perturb: mujoco.MjvPerturb,
  camera: mujoco.MjvCamera,
  gl_camera: mujoco.MjvGLCamera,
  extra_geoms: list[mujoco.MjvGeom],
  pixels: npt.NDArray[np.uint8],
) -> None:
  assert_type(mjrf.Camera(), mjrf.Camera)
  assert_type(mjrf.DrawMode.DRAW_MODE_DEPTH.value, int)
  assert_type(mjrf.PixelFormat.PIXEL_FORMAT_RGB8.name, str)
  assert_type(mjrf.LightType.LIGHT_SPOT.__index__(), int)

  config = mjrf.ContextConfig(
    graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL, software_rendering=False
  )
  config.software_rendering = True
  assert_type(config.software_rendering, bool)
  assert_type(config.graphics_api, mjrf.GraphicsApi)
  assert_type(config.native_window, int)
  cam = mjrf.Camera(pos=[0, 0, 1], forward=np.zeros(3))
  cam.up = [0.0, 1.0, 0.0]
  assert_type(cam.up, npt.NDArray[np.float32])
  assert_type(mjrf.Rect(width=32).height, int)
  assert_type(
    mjrf.VertexAttribute(
      usage=mjrf.VertexAttributeUsage.VERTEX_ATTRIBUTE_USAGE_POSITION,
      type=mjrf.VertexAttributeType.VERTEX_ATTRIBUTE_TYPE_FLOAT3,
    ).type,
    mjrf.VertexAttributeType,
  )

  mesh_config = mjrf.MeshConfig(
    attributes=[mjrf.VertexAttribute()],
    index_type=mjrf.IndexType.INDEX_TYPE_U16,
  )
  assert_type(mesh_config.attributes(0), mjrf.VertexAttribute)
  mesh_data = mjrf.MeshData(bounds_min=(0.0, 0.0, 0.0))
  assert_type(mesh_data.bounds_min, npt.NDArray[np.float32])
  mesh_data.set_vertices(mesh_config, b'')
  mesh_data.set_indices(mesh_config, b'')
  mesh = ctx.create_mesh(mesh_config)
  assert_type(mesh.upload(mesh_data), None)
  assert_type(ctx.create_mesh(mesh_config), mjrf.Mesh)
  texture = ctx.create_texture(mjrf.TextureConfig())
  assert_type(texture.upload(mjrf.TextureData()), None)
  assert_type(texture, mjrf.Texture)
  assert_type(ctx.create_scene(mjrf.SceneParams()), mjrf.Scene)
  light_params = mjrf.LightParams(
    color=[1, 1, 1], cast_shadow=True, cutoff=0.5, softness=0.1
  )
  assert_type(light_params.cast_shadow, bool)
  assert_type(light_params.cutoff, float)
  assert_type(light_params.softness, float)
  assert_type(ctx.create_light(light_params), mjrf.Light)
  render_params = mjrf.RenderableParams(cast_shadow=True, receive_shadow=False)
  assert_type(render_params.cast_shadow, bool)
  assert_type(render_params.receive_shadow, bool)
  assert_type(ctx.create_renderable(render_params), mjrf.Renderable)
  assert_type(
    ctx.create_render_target(mjrf.RenderTargetConfig()), mjrf.RenderTarget
  )

  material = mjrf.Material(color=(1, 1, 1, 1), decor=True)
  assert_type(material.decor, bool)
  material.sleep_state = mujoco.mjtSleepState.mjS_AWAKE
  assert_type(material.color, npt.NDArray[np.float32])
  assert_type(renderable.get_material(), mjrf.Material)
  renderable.set_pose([0, 0, 0], np.eye(3, dtype=np.float32).ravel())
  renderable.set_geom_mesh(mujoco.mjtGeom.mjGEOM_SPHERE, 1, 2, 3)
  assert_type(light.get_type(), int)
  light.set_active(True)
  light.set_cast_shadow(True)
  light.set_cutoff(0.5)
  light.set_pose([0, 0, 0], [0, 0, 1])
  scene.configure_from_model(model)
  assert_type(scene.set_msaa(4), None)

  read = mjrf.ReadRequest(target=target)
  read.set_buffer(memoryview(pixels))
  request = mjrf.RenderRequest(
    scene=scene,
    target=target,
    camera=cam,
    post_processing=True,
    reflections=True,
    shadows=True,
    headlight=True,
    headlight_color=[1.0, 1.0, 1.0],
    headlight_intensity=1.0,
  )
  request.shadows = False
  assert_type(request.headlight, bool)
  assert_type(request.headlight_color, npt.NDArray[np.float32])
  assert_type(request.headlight_intensity, float)
  assert_type(ctx.render([request], [read]), int)
  assert_type(ctx.get_frame_stats(0, mjrf.FrameStats()), None)

  resources = mjrf.ModelResources(ctx, model)
  renderables = mjrf.ModelRenderables(scene, resources)
  assert_type(renderables.update(data), None)
  assert_type(
    renderables.get_renderable(mujoco.mjtObj.mjOBJ_GEOM, 0),
    mjrf.Renderable | None,
  )
  lights = mjrf.ModelLights(scene, resources)
  assert_type(lights.update(data), None)
  assert_type(lights.get_light(0), mjrf.Light | None)
  model_view = model_scene.ModelScene(ctx, model, scene)
  assert_type(model_view.update(data), None)
  assert_type(model_view.scene, mjrf.Scene)
  assert_type(model_view.get_light(0), mjrf.Light | None)
  assert_type(
    model_view.get_renderable(mujoco.mjtObj.mjOBJ_GEOM, 0),
    mjrf.Renderable | None,
  )
  decorations = mjrf.ModelDecorations(ctx, scene, model)
  decorations.update(data, option, perturb, camera, mjrf.Rect(), extra_geoms)

  wrapper = renderer.Renderer(ctx)
  assert_type(wrapper.context, mjrf.Context)
  assert_type(wrapper.scene('scene'), mjrf.Scene)
  assert_type(wrapper.target('rgb', pixels), mjrf.RenderTarget)
  view = renderer.View(scene='scene', target='rgb', camera=gl_camera)
  assert_type(view.headlight_color, tuple[float, float, float])
  view.camera = cam
  assert_type(wrapper.render([view]), None)
