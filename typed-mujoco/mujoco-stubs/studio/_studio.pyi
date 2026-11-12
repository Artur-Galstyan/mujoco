# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Native Studio extension. Its submodules are importable from mujoco.studio."""

from mujoco import _render_filament as _render_filament
from mujoco.studio import native_viewer_cc as _native_viewer_cc
from mujoco.studio import sim as _sim
from mujoco.studio import ux as _ux
from mujoco.studio import window as _window
from mujoco.studio.web import headless_ui as _headless_ui
from mujoco.studio.web import state_payload as _state_payload

native_viewer_cc = _native_viewer_cc
sim = _sim
ux = _ux
window = _window
headless_ui = _headless_ui
state_payload = _state_payload
