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

"""Static studio app consumer; ty checks this file but it must never run."""

import types
from collections.abc import Iterable
from typing import Any

from typing_extensions import assert_type

import mujoco
from mujoco import studio
from mujoco.studio import app as studio_app
from mujoco.studio import (
  endpoints,
  hot_reload,
  hot_reload_utils,
  launch_native,
  launch_passive,
  launch_thread,
  launch_web,
  messages,
  parser,
  plugin,
  plugin_registry,
  sim,
  ux,
  viewer_handle,
  viewer_protocol,
)
from mujoco.studio.app import inputs, plugins


class MySim(plugin.SimPlugin):
  pass


class MyViewer(plugin.ViewerPlugin):
  pass


def package() -> None:
  assert_type(studio.__version__, str)
  assert_type(studio.ux.GuiTheme.DARK.value, int)
  assert_type(studio_app.Config(), plugins.Config)
  assert_type(studio_app.SIM_PLUGINS, tuple[plugin.SimPlugin, ...])
  assert_type(studio_app.plugins.SIM_PLUGINS, tuple[plugin.SimPlugin, ...])
  studio_app.install_console_hooks()


def app_plugin(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  config = studio_app.Config(show_info=True, show_console=True)
  assert_type(config.show_info, bool)
  assert_type(config.show_inspector, bool)
  app = studio_app.App(config)
  assert_type(app, plugins.App)
  assert_type(app.config, plugins.Config)
  assert_type(app.status, str)
  assert_type(app.theme, ux.GuiTheme)
  assert_type(app.step_control_state, sim.StepControl)
  assert_type(app.ux_state, ux.UxState)
  assert_type(app.model, mujoco.MjModel)
  assert_type(app.data, mujoco.MjData)
  assert_type(app.model_path, str)
  assert_type(app.viewer, viewer_protocol.Viewer)
  assert_type(app.is_running(), bool)
  app.model = model
  app.data = data
  app.update()
  app.build_gui()
  app.reset_physics()
  app.close()
  event = studio_app.InitEvent(app=app)
  assert_type(event.app, plugins.App)
  assert isinstance(app, plugin.ViewerPlugin)


def keyboard(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  camera: mujoco.MjvCamera,
  vis: mujoco.MjvOption,
  perturb: mujoco.MjvPerturb,
  step: sim.StepControl,
  state: ux.UxState,
  app: plugins.App,
) -> None:
  assert_type(inputs.handle_reset_keyboard_events(model, data), bool)
  assert_type(
    inputs.handle_keyboard_events(model, data, camera, vis, step, state, 1.0),
    tuple[bool, float],
  )
  assert_type(
    inputs.handle_freecam_wasd_keyboard_events(model, data, camera, 0.5),
    tuple[bool, float],
  )
  assert_type(inputs.handle_miscellaneous_keyboard_events(app), bool)
  assert_type(inputs.handle_vis_options_keyboard_events(vis, False), bool)
  assert_type(inputs.handle_step_control_keyboard_events(step, state), bool)
  inputs.handle_mouse_events(model, data, camera, vis, perturb, state)
  inputs.handle_mouse_events(
    model, data, camera, vis, perturb, state, move_camera=False
  )


def endpoint_pair() -> None:
  viewer_end, sim_end = launch_thread.make_thread_endpoints()
  assert_type(viewer_end, endpoints.ViewerEndpoint)
  assert_type(sim_end, endpoints.SimEndpoint)
  assert_type(sim_end.get_viewer_events(), list[messages.Event])
  assert_type(viewer_end.get_sim_snapshots(), list[messages.Snapshot])
  assert_type(
    sim_end.reclaim_to_viewer(messages.Snapshot), messages.Snapshot | None
  )
  channel = launch_thread.ThreadEventChannel()
  assert_type(channel.get(), list[messages.Event])
  snapshots = launch_thread.ThreadSnapshotChannel()
  assert_type(snapshots.get(), list[messages.Snapshot])
  pair = endpoints.make_endpoints(
    s2v_snapshot=snapshots,
    s2v_events=channel,
    v2s_events=launch_thread.ThreadEventChannel(),
    v2s_snapshot=launch_thread.ThreadSnapshotChannel(),
  )
  assert_type(pair, tuple[endpoints.ViewerEndpoint, endpoints.SimEndpoint])
  viewer_end.close()


def launchers(
  config: viewer_protocol.ViewerConfig, model: mujoco.MjModel, path: str
) -> None:
  extra: Iterable[plugin.Plugin] = [MySim(), MyViewer()]
  assert_type(
    launch_passive.launch(config, plugins=extra), viewer_handle.ViewerHandle
  )
  assert_type(launch_native.launch(), viewer_handle.ViewerHandle)
  assert_type(
    launch_web.launch(plugins=[plugins.App()]), viewer_handle.ViewerHandle
  )
  launch_passive.run(config, model=model, plugins=[*studio_app.SIM_PLUGINS])
  launch_native.run(model_path=path)
  launch_web.run(config, model=model)
  handle = launch_thread.launch_thread(
    lambda endpoint: None, sim_plugins=[MySim()]
  )
  assert_type(handle, viewer_handle.ViewerHandle)
  viewer_plugins, sim_plugins = plugin.split_plugins(extra)
  assert_type(viewer_plugins, list[plugin.ViewerPlugin])
  assert_type(sim_plugins, list[plugin.SimPlugin])
  assert_type(plugin.load_plugins([path]), list[plugin.Plugin])
  assert_type(plugin.PLUGIN_PACKAGE, str)


def model_files(path: str) -> None:
  assert_type(parser.load(path), mujoco.MjModel)
  assert_type(parser.parse(path), mujoco.MjData)
  assert_type(parser.dependencies(path), list[str])


def registry() -> None:
  reg = plugin_registry.PluginRegistry([MySim()])
  assert_type(reg.plugins, list[Any])
  reg.rebuild_handlers()
  reg.dispatch(messages.ExitEvent())
  prioritized = plugin_registry.PrioritizedHandler(10, print)
  assert_type(prioritized.priority, int)


def reloading(path: str, reg: plugin_registry.PluginRegistry) -> None:
  options = hot_reload.Options.parse('model,MyViewer')
  assert_type(options, hot_reload.Options)
  assert_type(options.model, bool)
  assert_type(options.plugins, bool | frozenset[str])
  assert_type(options.reloads(MyViewer()), bool)
  reloader = hot_reload.Reloader(options, poll_interval=0.1)
  assert_type(reloader.generation, int)
  assert_type(reloader.watched_modules, set[str])
  assert_type(reloader.poll_interval, float)
  hot_reload.enable('plugins')
  assert_type(hot_reload.watches_model(), bool)
  watcher = hot_reload.watch_files([path])
  assert_type(watcher, hot_reload_utils.FileWatcher | None)
  if watcher is not None:
    assert_type(watcher.poll(), set[str])
    assert_type(watcher.watch(path), str)


def reload_helpers(module: types.ModuleType) -> None:
  assert_type(hot_reload_utils.source_path(module), str | None)
  assert_type(hot_reload_utils.local_imports(module), list[types.ModuleType])
  assert_type(
    hot_reload_utils.reload_order({'a': {'b'}, 'b': set()}, ['b']), list[str]
  )
  assert_type(hot_reload_utils.class_name(MySim), str)
  assert_type(hot_reload_utils.same_default(1, 1), bool)
  hot_reload_utils.reload_module(module)


class Counting(plugin.ViewerPlugin):
  count: int = 0

  @messages.handler(priority=messages.Priority.CRITICAL)
  def on_update(self, _: messages.UpdateEvent) -> None:
    self.count += 1
    assert_type(self.viewer, viewer_protocol.Viewer)


def plugin_hosts(
  handle: viewer_handle.ViewerHandle,
  host: viewer_protocol.Viewer,
  reg: plugin_registry.PluginRegistry,
) -> None:
  sim_plugin = MySim()
  sim_plugin.handle = handle
  sim_plugin.handle = None
  assert_type(sim_plugin.handle, viewer_handle.ViewerHandle)
  viewer_plugin = Counting()
  viewer_plugin.viewer = None
  assert_type(viewer_plugin.viewer, viewer_protocol.Viewer)
  init = plugin.ViewerInitEvent(viewer=host)
  assert_type(init.viewer, viewer_protocol.Viewer)
  assert_type(
    plugin.SimInitEvent(handle=handle).handle, viewer_handle.ViewerHandle
  )
  assert_type(plugin.ThemeEvent(theme=ux.GuiTheme.DARK).theme, ux.GuiTheme)
  hot_reload.tick(reg, host)
  hot_reload.tick(reg, handle)
  hot_reload.Reloader().tick(reg, host)
  assert_type(hot_reload_utils.current_class(MySim), type[Any] | None)
  assert_type(hot_reload_utils.module_dir('os'), str | None)
  assert_type(plugin.load_plugins(), list[plugin.Plugin])
