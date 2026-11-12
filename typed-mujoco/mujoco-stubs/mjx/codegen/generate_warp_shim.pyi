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

"""Creates a shim between JAX and Warp for a given function."""

from collections.abc import Sequence
from typing import Dict

from etils import epath

from mujoco.mjx.codegen import trace

def create_jax_warp_shim(
  fn_name: str,
  field_usage: trace.FieldUsage,
  mjwarp_field_info: Dict[str, trace.FieldInfo],
  mjx_warp_field_info: Dict[str, trace.FieldInfo],
  out_fpath: epath.Path,
) -> None:
  """Creates a JAX-wrapped MJWarp function."""

def main(argv: Sequence[str]) -> None: ...
