# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Static Studio viewer consumer; ty checks this file but it must never run."""

import contextlib
import queue
import socket
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco.studio import (
  endpoints,
  messages,
  native_viewer,
  plugin,
  sim,
  sim_history,
  step_control,
  ux,
  viewer,
  viewer_handle,
  viewer_protocol,
  viewer_utils,
  web_viewer,
)
from mujoco.studio.web import (
  client_camera,
  headless_ui,
  state_payload,
  web_server,
)
from mujoco.studio.web import headless_ui as web_headless_ui


def messages_api(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  assert_type(messages.KEYFRAME_STATE_SIG, int)
  state = np.zeros(3, dtype=np.float64)
  snap = messages.StateSnapshot(state=state, state_sig=1)
  assert_type(snap.state, npt.NDArray[np.float64])
  assert_type(snap.state_sig, int)
  assert_type(snap == snap, bool)
  event = messages.StateEvent.from_data(model, data, 1)
  assert_type(event, messages.StateEvent | None)
  frame = messages.KeyframeEvent.from_model(model, 0)
  assert_type(frame, messages.KeyframeEvent | None)
  assert_type(messages.ModelEvent(model).path, str)
  assert_type(messages.ResetEvent().key, int | None)
  assert_type(
    messages.RequestPauseEvent(sim.PauseState.UNPAUSED).pause_state,
    sim.PauseState,
  )
  step = messages.StepControlSnapshot.from_step_control(sim.StepControl(), 1.0)
  assert_type(step, messages.StepControlSnapshot)
  assert_type(step.speed, float)
  assert_type(messages.describe_exception(None), str)
  assert_type(int(messages.Priority.USER), int)
  assert_type(messages.HotReloadEvent(plugins=()).plugins, tuple[Any, ...])
  assert_type(messages.PostStepEvent(model, data).data, mujoco.MjData)

  def make() -> messages.Message | None:
    return messages.UpdateEvent()

  maker: messages.MakeMessage = make
  sent: list[messages.Message] = []
  with messages.send_if_changes(sent.append, maker, should_send=lambda: True):
    pass

  class Handlers:
    @messages.handler
    def on_reset(self, event: messages.ResetEvent) -> bool:
      return event.key is None

    @messages.handler(priority=messages.Priority.CRITICAL)
    def on_exit(self, event: messages.ExitEvent) -> None:
      del event

  assert_type(Handlers().on_reset(messages.ResetEvent()), bool)

  class Channel:
    def put(self, value: messages.Event) -> None:
      del value

    def get(self) -> list[messages.Event]:
      return []

    def close(self) -> None:
      pass

  channel: messages.EventChannel = Channel()
  assert_type(channel.get(), list[messages.Event])


def protocol_api(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  view: viewer_protocol.Viewer,
  config: viewer_protocol.ViewerConfig,
) -> None:
  assert_type(viewer_protocol.GFX_MODES[0], Literal['opengl'])
  assert_type(config.gfx, str)
  assert_type(config.http_port, int)
  assert_type(view.model, mujoco.MjModel)
  assert_type(view.camera, mujoco.MjvCamera)
  assert_type(view.render_flags, ux.RenderFlags)
  assert_type(view.client_camera, client_camera.ClientCameraSync | None)
  assert_type(view.extra_geoms, list[mujoco.MjvGeom])
  assert_type(view.is_running(), bool)
  assert_type(view.get_sim_events(), list[messages.Event])
  assert_type(view.get_sim_snapshots(), list[messages.Snapshot])
  assert_type(view.send_if_changes(), contextlib.AbstractContextManager[None])
  assert_type(view.prepare_next_frame(), bool)
  assert_type(view.get_drop_file(), str)
  assert_type(view.upload_image(0, b'', 1, 1, 4), int)
  view.send_to_sim(messages.SingleStepEvent())
  view.dispatch(messages.ThreadPoolEvent(nthread=2))
  view.load_model(model, 'a.xml')
  assert_type(
    viewer_protocol.ViewerInitEvent(view).viewer, viewer_protocol.Viewer
  )
  assert_type(viewer_protocol.ThemeEvent(ux.GuiTheme.DARK).theme, ux.GuiTheme)
  viewer_protocol.run_viewer_loop(view)
  viewer_utils.send_state(view, model, data, 1)
  viewer_utils.apply_perturb(view, None, None)
  viewer_utils.apply_perturb(view, model, data, is_paused=True)
  assert_type(viewer.main(), None)
  assert_type(viewer.vp.ViewerConfig(), viewer_protocol.ViewerConfig)
  assert_type(viewer_utils.send_state(view, model, data, 1), None)


def native_and_web(
  endpoint: endpoints.ViewerEndpoint, model: mujoco.MjModel
) -> None:
  config = viewer_protocol.ViewerConfig(title='t', gfx='web')
  native = native_viewer.NativeViewer(config, endpoint, model=model)
  assert_type(native, native_viewer.NativeViewer)
  assert_type(native.get_drop_file(), str)
  web = web_viewer.WebViewer(config, endpoint, plugins=[])
  assert_type(web.port, int)
  assert_type(web.get_drop_file(), str)
  web.sync_idle()
  viewer_protocol.run_viewer_loop(web)


def handle_api(
  sim_endpoint: endpoints.SimEndpoint, model: mujoco.MjModel
) -> None:

  def alive() -> bool:
    return True

  is_alive: viewer_handle.IsAliveFn = alive
  with viewer_handle.ViewerHandle(sim_endpoint, is_alive_fn=is_alive) as handle:
    assert_type(handle, viewer_handle.ViewerHandle)
    assert_type(handle.model, mujoco.MjModel | None)
    assert_type(handle.model_path, str)
    handle.set_model(mujoco.MjSpec(), path='a.xml', watch=['b.xml'])
    handle.reload()
    out = handle.sync()
    assert_type(out, tuple[mujoco.MjModel | None, mujoco.MjData | None])
    assert_type(handle.is_running(), bool)
    assert_type(
      handle.send_if_changes(), contextlib.AbstractContextManager[None]
    )
    handle.send_to_viewer(messages.ResetEvent(key=0))
    viewer_handle.run_sim_loop(handle)
    init = viewer_handle.SimInitEvent(handle=handle)
    assert_type(init.handle, viewer_handle.ViewerHandle)

  def fn(timeout: float) -> None:
    del timeout

  def err() -> None:
    pass

  shutdown: viewer_handle.ShutdownFn = fn
  forward: viewer_handle.ForwardErrorFn = err
  viewer_handle.ViewerHandle(
    sim_endpoint, shutdown_fn=shutdown, forward_error_fn=forward
  )


def sim_plugins() -> None:
  history = sim_history.SimHistory(min_seconds=2.0)
  assert_type(history, sim_history.SimHistory)
  assert_type(history.sim_history, sim.SimHistory)
  assert_type(history.frame_interval, float)
  assert_type(history.handle, viewer_handle.ViewerHandle)
  scrub = sim_history.SimHistorySnapshot.from_ux_state(ux.UxState())
  assert_type(scrub, sim_history.SimHistorySnapshot | None)
  assert_type(sim_history.SimHistorySnapshot().size, int)
  assert_type(sim_history.SimHistorySnapshot().head_time, float)
  control = step_control.StepControl()
  assert_type(control.step_control, sim.StepControl)
  plugins: list[plugin.SimPlugin] = [history, control]
  assert_type(plugins[0].handle, viewer_handle.ViewerHandle)


def web_api() -> None:
  client = client_camera.ClientCameraSync()
  assert_type(client.generation, int)
  camera = mujoco.MjvCamera()
  assert_type(client.apply(camera, b''), bool)
  assert_type(client.update_generation(camera), int)

  sock = web_server.bind_public_socket('::', 0)
  assert_type(sock, socket.socket)
  ui = headless_ui.HeadlessUi('title')
  drops: queue.Queue[dict[str, bytes]] = queue.Queue()
  server = web_server.WebServer(sock, ui=ui, mjb_data=b'', drop_queue=drops)
  session: web_server.UiSession = ui
  assert_type(session.take_messages(), list[bytes])
  assert_type(server.ui, web_server.UiSession | None)
  assert_type(server.http_sock, socket.socket)
  assert_type(server.static_files_dir, str | None)
  assert_type(server.viewer_count, int)
  assert_type(server.models_generation, int)
  assert_type(server.add_model(b'x'), int)
  assert_type(server.cached_model(1), bytes | None)
  assert_type(server.prefetch_line(), str)
  assert_type(server.pop_theme(), str | None)
  assert_type(server.take_client_camera(), bytes | None)
  server.update_state(b'')
  server.update_model(b'', crc32=3)
  server.set_theme('dark')
  server.set_client_camera(b'')
  server.notify_ui_frame()
  server.start()
  server.stop(reason='x')
  assert_type(state_payload.MAX_EXTRA_GEOMS, int)
  assert_type(web_headless_ui.HeadlessUi('t'), headless_ui.HeadlessUi)
  callback: Callable[[], None] = server.start
  callback()
