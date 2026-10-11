# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Static Studio consumer; ty checks this file but it must never run."""

import numpy as np
from typing_extensions import Never, assert_type

import mujoco
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.implot import implot
from mujoco.experimental.studio import (
  native_viewer_cc,
  renderer,
  sim,
  ux,
  window,
)
from mujoco.experimental.studio.web import headless_ui, state_payload


def studio_native(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  camera = mujoco.MjvCamera()
  perturb = mujoco.MjvPerturb()
  vis_options = mujoco.MjvOption()
  geom = mujoco.MjvGeom()
  flags = ux.RenderFlags()
  flags.flags = range(11)
  flags.flags = np.zeros(11, dtype=np.uint8)
  assert_type(flags.flags, list[int])

  state = ux.UxState()
  state.state = [0.0, 1.0]
  state.state = np.zeros(2, dtype=np.float64)
  assert_type(state.state, list[float])
  assert_type(state.history_index, int)
  assert_type(ux.configure_docking_layout(), tuple[float, float, float, float])
  assert_type(ux.GuiTheme.DARK.value, int)
  assert_type(ux.GuiTheme(ux.GuiTheme.DARK.value).name, str)
  assert_type(ux.CameraMotion(ux.CameraMotion.ORBIT.value), ux.CameraMotion)
  assert_type(sim.StepStatus.PAUSED.value, int)
  assert_type(sim.StepStatus(sim.StepStatus.PAUSED.value).name, str)
  assert_type(sim.PauseState(sim.PauseState.UNPAUSED.value), sim.PauseState)
  assert_type(ux.load_theme(def_theme=ux.GuiTheme.DARK), ux.GuiTheme)
  assert_type(ux.theme_menu_gui(ux.GuiTheme.LIGHT), tuple[bool, ux.GuiTheme])
  assert_type(ux.set_camera(model, camera, -1), int)
  ux.MoveCamera(model, data, camera, ux.CameraMotion.ORBIT, 0.1, 0.2)
  pick = ux.Pick(model, data, camera, 0.5, 0.5, 1.0, vis_options)
  assert_type(pick, ux.PickResult)
  assert_type(pick.point, tuple[float, float, float])
  pick.point = (0.0, 1.0, 2.0)
  assert_type(
    ux.pip_gui([ux.PipSource('cam', 1)], [ux.PipState()]), list[ux.PipState]
  )

  control = sim.StepControl()
  assert_type(control.advance(model, data), sim.StepStatus)
  assert_type(control.advance(None, None), sim.StepStatus)
  assert_type(control.get_pause_state(), sim.PauseState)
  control.set_pause_state(sim.PauseState.NORMAL_PAUSED)
  assert_type(control.get_noise_parameters(), tuple[float, float])
  history = sim.SimHistory()
  history.init(3, max_history=16)
  assert_type(history.size(), int)
  ux.reset_history(history, state, model, data)

  studio = native_viewer_cc.Viewer('Studio', 320, 240, 'opengl')
  studio.InitRenderer(model)
  studio.Present(model, data, perturb, camera, vis_options, flags.flags, [geom])
  studio.Present(
    model,
    data,
    perturb,
    camera,
    vis_options,
    np.zeros(11, dtype=np.uint8),
    (geom,),
  )
  assert_type(studio.RenderToTexture(model, data, camera, 64, 64, 0), int)
  assert_type(studio.UploadImage(0, b'\x00' * 3, 1, 1, 3), int)
  assert_type(studio.NewFrame(), bool)
  assert_type(studio.GetImPlotContext(), int)

  render = renderer.Renderer('opengl_headless')
  render.Init(model)
  assert_type(render.Render(model, data, None, None, None, 32, 32), bytes)
  assert_type(render.get_render_flags(), memoryview)
  win = window.Window('Studio', 320, 240, 'opengl_headless')
  win.Present(np.zeros((240, 320, 3), dtype=np.uint8))
  win.Present(memoryview(bytearray(3)))
  assert_type(win.GetWidth(), int)

  web = headless_ui.HeadlessUi('Studio')
  assert_type(web.new_frame(), bool)
  assert_type(web.get_context(), int)
  assert_type(web.upload_image(0, b'\x00' * 4, 1, 1, 4), int)
  assert_type(
    state_payload.serialize_state_payload(
      0,
      0,
      b'\x00',
      camera,
      perturb,
      vis_options,
      model,
      np.zeros(11, dtype=np.uint8),
      (geom,),
    ),
    bytes,
  )
  assert_type(state_payload.max_state_payload_size(8), int)
  assert_type(state_payload.MAX_EXTRA_GEOMS, int)


def studio_ui() -> None:
  point = imgui.Vec2(1.0, 2.0)
  assert_type(point.x, float)
  assert_type(imgui.GetIO().DisplaySize, imgui.Vec2)
  assert_type(imgui.Begin('Options', True), tuple[bool, bool])
  assert_type(imgui.Key.F6.value, int)
  assert_type(imgui.Key(imgui.Key.F6.value).name, str)
  assert_type(imgui.WindowFlags.__members__['None'], imgui.WindowFlags)
  assert_type(imgui.MenuItem('File'), bool | tuple[bool, bool])
  assert_type(imgui.MenuItem('File', 'Ctrl+F', True), bool | tuple[bool, bool])
  assert_type(imgui.BeginTooltip(), bool)
  assert_type(imgui.BeginItemTooltip(), bool)
  assert_type(imgui.TextLinkOpenURL('Docs'), bool)
  assert_type(imgui.OpenPopup('Menu'), bool)
  assert_type(imgui.OpenPopup(1), bool)
  assert_type(imgui.OpenPopupOnItemClick(), bool)
  assert_type(imgui.BeginTabItem('Tab'), bool)
  assert_type(imgui.BeginTabItem('Tab', True), tuple[bool, bool])
  assert_type(imgui.TableSetColumnIndex(0), bool)
  assert_type(imgui.Button('OK', point), bool)
  assert_type(imgui.IsKeyChordPressed(imgui.Key.F6), bool)
  assert_type(imgui.InputFloat('Speed', 1.0), tuple[bool, float])
  assert_type(
    imgui.ColorEdit4('Tint', [0.0, 0.0, 0.0, 1.0]), tuple[bool, list[float]]
  )
  assert_type(imgui.ListClipper().Step(), bool)
  assert_type(implot.Flags.NoInputs.value, int)
  assert_type(implot.Flags(implot.Flags.NoInputs.value).name, str)
  assert_type(implot.Flags.__members__['None'], implot.Flags)
  assert_type(implot.Axis(implot.Axis.X1.value), implot.Axis)
  assert_type(implot.Point(1.0, 2.0).x, float)
  assert_type(
    implot.BeginPlot('Position', point, flags=implot.Flags.NoInputs), bool
  )
  implot.SetupAxis(implot.Axis.X1, 'time', implot.AxisFlags.NoGridLines)
  implot.SetupAxisTicks(implot.Axis.X1, range(3), ['0', '1', '2'])
  implot.PlotLine('position', range(3), np.arange(3, dtype=np.float64))
  assert_type(implot.PixelsToPlot(point), implot.Point)
  assert_type(implot.PlotToPixels(0.0, 0.0), imgui.Vec2)


def unbound_implot_rect(which: bool) -> None:
  if which:
    assert_type(implot.GetPlotLimits(), Never)
  else:
    assert_type(implot.GetPlotSelection(), Never)
