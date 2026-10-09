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

"""Check types as a consumer of the installed wheel, never the source tree."""

import ast
import importlib
import importlib.metadata
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = Path(__file__).resolve().parents[2]
NATIVE = (
  '_callbacks',
  '_constants',
  '_enums',
  '_errors',
  '_functions',
  '_render',
  '_rollout',
  '_simulate',
  '_specs',
  '_structs',
)
STUBS = (
  '__init__.pyi',
  *(f'{name}.pyi' for name in NATIVE),
  'gl_context.pyi',
  'renderer.pyi',
  'rollout.pyi',
  'viewer.pyi',
  'rendering/classic/gl_context.pyi',
  'rendering/classic/renderer.pyi',
)
ENV = {
  key: value
  for key, value in os.environ.items()
  if key.upper()
  not in {
    'PYTHONPATH',
    'PYTHONHOME',
    'XDG_CONFIG_HOME',
    'APPDATA',
    'LOCALAPPDATA',
  }
  and not key.upper().startswith('TY_')
}


def command(args: list[str], cwd: Path, expected: int = 0) -> str:
  result = subprocess.run(
    args, cwd=cwd, env=ENV, text=True, capture_output=True, timeout=60
  )
  output = result.stdout + result.stderr
  if result.returncode != expected:
    raise AssertionError(f'{args!r} returned {result.returncode}:\n{output}')
  return output


def installed_package(cwd: Path) -> Path:
  distribution = importlib.metadata.distribution('mujoco')
  assert distribution.files is not None, 'mujoco lacks installed file metadata'
  recorded = {file.as_posix() for file in distribution.files}
  required = {'mujoco/py.typed', *(f'mujoco/{stub}' for stub in STUBS)}
  assert not required - recorded, f'Wheel lacks: {sorted(required - recorded)}'
  package = Path(str(distribution.locate_file('mujoco'))).resolve()
  assert not package.is_relative_to(PYTHON), 'mujoco resolves to the checkout'
  assert all((package / stub).is_file() for stub in STUBS)
  assert (package / 'py.typed').is_file()
  spec = importlib.util.find_spec('mujoco')
  assert (
    spec
    and spec.origin
    and Path(spec.origin).resolve() == package / '__init__.py'
  )
  origin = command(
    [
      sys.executable,
      '-I',
      '-c',
      'import importlib.util; print(importlib.util.find_spec("mujoco").origin)',
    ],
    cwd,
  ).strip()
  assert Path(origin).resolve() == package / '__init__.py', origin
  return package


def declared_names(nodes: list[ast.stmt]) -> set[str]:
  names = {
    node.name
    for node in nodes
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
  }
  names.update(
    node.target.id
    for node in nodes
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
  )
  names.update(
    target.id
    for node in nodes
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name)
  )
  return names


def check_class_api(name: str, node: ast.ClassDef, runtime: type) -> None:
  label = f'{name}.{node.name}'
  actual = {key for key in vars(runtime) if not key.startswith('_')}
  if label == '_specs.MjVisual':
    actual.discard('global')  # Python cannot spell this keyword as a field.
  missing = actual - declared_names(node.body)
  assert not missing, f'{label}: missing members: {sorted(missing)}'
  setters = {
    member.name
    for member in node.body
    if isinstance(member, ast.FunctionDef)
    if any(
      isinstance(d, ast.Attribute) and d.attr == 'setter'
      for d in member.decorator_list
    )
  }
  for member in node.body:
    if isinstance(member, ast.ClassDef):
      child = vars(runtime).get(member.name)
      assert isinstance(child, type), f'{label}.{member.name}: no runtime class'
      check_class_api(label, member, child)
    if isinstance(member, ast.FunctionDef) and any(
      isinstance(d, ast.Name) and d.id == 'property'
      for d in member.decorator_list
    ):
      prop = vars(runtime).get(member.name)
      if isinstance(prop, property):
        assert (member.name in setters) == (prop.fset is not None), (
          f'{label}.{member.name}: wrong setter flag'
        )


def check_native_api(package: Path) -> None:
  modules = {}
  for name in NATIVE:
    runtime = importlib.import_module(f'mujoco.{name}')
    modules[name] = runtime
    tree = ast.parse((package / f'{name}.pyi').read_text())
    actual = {
      key
      for key in vars(runtime)
      if not key.startswith('_') or key.startswith(('_MjModel', '_MjData'))
    }
    missing = actual - declared_names(tree.body)
    assert not missing, f'{name}: missing names: {sorted(missing)}'
    for node in tree.body:
      if isinstance(node, ast.ClassDef):
        check_class_api(name, node, getattr(runtime, node.name))

  import mujoco

  root = ast.parse((package / '__init__.pyi').read_text())
  declared = declared_names(root.body)
  declared.update(
    alias.asname or alias.name
    for node in root.body
    if isinstance(node, ast.ImportFrom)
    for alias in node.names
  )
  native = set().union(
    *(
      {key for key in vars(module) if not key.startswith('_')}
      for name, module in modules.items()
      if name not in {'_simulate', '_rollout'}
    )
  )
  missing = native & vars(mujoco).keys() - declared
  assert not missing, f'mujoco: missing root exports: {sorted(missing)}'


def check_typing(cwd: Path, package: Path) -> None:
  ENV.update(XDG_CONFIG_HOME=str(cwd), APPDATA=str(cwd), LOCALAPPDATA=str(cwd))
  config = cwd / 'ty.toml'
  config.write_text('')
  ty = shutil.which('ty')
  ruff = shutil.which('ruff')
  assert ty and ruff, 'ty and ruff must be on PATH'
  stubs = [package / stub for stub in STUBS]
  local_stubs = [PYTHON / 'mujoco' / stub for stub in STUBS]
  generator = PYTHON / 'mujoco/codegen/generate_stubs.py'
  assert generator.is_file() and all(p.is_file() for p in local_stubs), (
    'Checked-in generator and stubs are required'
  )
  consumer = cwd / 'consumer.py'
  shutil.copyfile(Path(__file__).with_name('consumer.py'), consumer)
  ty_cmd = [
    ty,
    'check',
    '--config-file',
    str(config),
    '--no-respect-ignore-files',
    '--project',
    str(cwd),
    '--python',
    sys.executable,
    '--no-progress',
    '--output-format',
    'concise',
    '--error-on-warning',
  ]
  command(
    [
      ruff,
      'check',
      '--config',
      str(PYTHON / 'pyproject.toml'),
      str(generator),
      str(Path(__file__)),
      str(Path(__file__).with_name('consumer.py')),
      *(str(p) for p in local_stubs),
    ],
    cwd,
  )
  command(
    [
      ruff,
      'format',
      '--check',
      '--config',
      str(PYTHON / 'pyproject.toml'),
      str(generator),
      str(Path(__file__)),
      str(Path(__file__).with_name('consumer.py')),
      *(str(p) for p in local_stubs),
    ],
    cwd,
  )
  command([*ty_cmd, *(str(p) for p in stubs)], cwd)
  command([*ty_cmd, str(consumer)], cwd)
  command(
    [*ty_cmd, '--python-version', '3.11', str(Path(__file__)), str(generator)],
    cwd,
  )

  prefix = (
    'import mujoco\nimport numpy as np\n\n'
    'model = mujoco.MjModel.from_xml_string("<mujoco/>")\n'
    'data = mujoco.MjData(model)\nspec = mujoco.MjSpec()\n'
  )
  cases = {
    'constructor': ("mujoco.MjData('not a model')", 'invalid-argument-type'),
    'step': ("mujoco.mj_step(model, 'not data')", 'invalid-argument-type'),
    'nstep': (
      "mujoco.mj_step(model, data, nstep='no')",
      'invalid-argument-type',
    ),
    'field': ("data.time = 'not a number'", 'invalid-assignment'),
    'missing_field': ('model.geom(0).not_a_geom_field', 'unresolved-attribute'),
    'missing_option': ('model.opt.not_an_option', 'unresolved-attribute'),
    'missing_export': ('mujoco.not_a_real_export', 'unresolved-attribute'),
    'spec_keyword': (
      'spec.worldbody.add_body(not_a_field=3)',
      'unknown-argument',
    ),
    'spec_value': (
      "spec.worldbody.add_body(pos='bad')",
      'invalid-argument-type',
    ),
    'freejoint_keyword': (
      'spec.worldbody.add_freejoint(not_a_field=3)',
      'unknown-argument',
    ),
    'time_callback': (
      'mujoco.set_mjcb_time(lambda model: 0.0)',
      'invalid-argument-type',
    ),
    'control_callback': (
      'mujoco.set_mjcb_control(lambda: None)',
      'invalid-argument-type',
    ),
    'malloc_callback': (
      'mujoco.set_mju_user_malloc(lambda size: bytearray(size))',
      'invalid-argument-type',
    ),
    'jac_list': (
      'mujoco.mj_jac(model, data, [], np.zeros((3, model.nv)), np.zeros(3), 0)',
      'invalid-argument-type',
    ),
    'jac_dtype': (
      'mujoco.mj_jac(model, data, np.zeros((3, model.nv), dtype=np.int32), np.zeros((3, model.nv)), np.zeros(3), 0)',
      'invalid-argument-type',
    ),
  }
  line = len(prefix.splitlines()) + 1
  for name, (bad_code, rule) in cases.items():
    negative = cwd / f'{name}.py'
    negative.write_text(prefix + bad_code + '\n')
    output = command([*ty_cmd, str(negative)], cwd, expected=1)
    diagnostics = re.findall(
      r'(?m)^.*?:(\d+):\d+: (?:error|warning)\[([^]]+)\]', output
    )
    assert diagnostics == [(str(line), rule)], f'{name}: {output}'
    negative.unlink()


def smoke() -> None:
  import mujoco

  xml = (
    '<mujoco><worldbody><body name="body">'
    '<freejoint name="joint"/>'
    '<geom name="ball" type="sphere" size=".1"/>'
    '</body></worldbody></mujoco>'
  )
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_step(model, data)
  assert data.time > 0
  spec = mujoco.MjSpec.from_string(xml)
  compiled = spec.compile()
  body = spec.body('body')
  assert body is not None and compiled.bind(body).name == 'body'
  assert mujoco.MjData(compiled).bind(body).xpos.shape == (3,)
  archive = io.BytesIO()
  spec.to_zip(archive)
  archive.seek(0)
  assert mujoco.MjSpec.from_zip(archive).compile().nbody == compiled.nbody


def main() -> None:
  with tempfile.TemporaryDirectory() as directory:
    cwd = Path(directory)
    package = installed_package(cwd)
    check_native_api(package)
    smoke()
    check_typing(cwd, package)
  print('Installed wheel, runtime, native API, Ruff, and ty checks passed.')


if __name__ == '__main__':
  main()
