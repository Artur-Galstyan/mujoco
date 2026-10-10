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
import doctest
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
CORE_STUBS = (
  '__init__.pyi',
  *(f'{name}.pyi' for name in NATIVE),
  'gl_context.pyi',
  'renderer.pyi',
  'rollout.pyi',
  'viewer.pyi',
  'rendering/classic/gl_context.pyi',
  'rendering/classic/renderer.pyi',
)
OPTIONAL_STUBS = (
  '_render_filament.pyi',
  'mjrf.pyi',
  'rendering/filament/__init__.pyi',
  'rendering/filament/renderer.pyi',
  'experimental/studio/native_viewer_cc.pyi',
  'experimental/studio/renderer.pyi',
  'experimental/studio/ux.pyi',
  'experimental/studio/sim.pyi',
  'experimental/studio/window.pyi',
  'experimental/studio/web/headless_ui.pyi',
  'experimental/studio/web/state_payload.pyi',
  'experimental/dear_imgui/dear_imgui.pyi',
  'experimental/implot/implot.pyi',
)
STUBS = (*CORE_STUBS, *OPTIONAL_STUBS)
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


def check_core_annotations(root: Path) -> None:
  # std::byte has no safe Python scalar type in the MjByteVec binding.
  assert (
    'DefineArray<std::byte>(m, "MjByteVec");'
    in (PYTHON / 'mujoco/specs.cc').read_text()
  ), 'Review MjByteVec Any slots after a binding change'
  allowed = [
    ('_specs.pyi', 'MjByteVec', '__getitem__', 'return'),
    ('_specs.pyi', 'MjByteVec', '__init__', 'arg0'),
    ('_specs.pyi', 'MjByteVec', '__iter__', 'return'),
    ('_specs.pyi', 'MjByteVec', '__setitem__', 'arg1'),
  ]
  found = []
  for stub in CORE_STUBS:
    tree = ast.parse((root / stub).read_text(), filename=stub)
    parents = {
      child: parent
      for parent in ast.walk(tree)
      for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        assert node.returns is not None and not any(
          isinstance(part, ast.Name) and part.id == 'object'
          for part in ast.walk(node.returns)
        ), f'{stub}:{node.lineno}: bare broad return type'
      if isinstance(node, ast.Name) and node.id == 'Unknown':
        raise AssertionError(f'{stub}:{node.lineno}: Unknown type')
      if not (
        isinstance(node, ast.Name)
        and node.id == 'Any'
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'typing'
        and node.attr == 'Any'
      ):
        continue
      ancestors = []
      parent = node
      while parent in parents:
        parent = parents[parent]
        ancestors.append(parent)
      cls = next((p for p in ancestors if isinstance(p, ast.ClassDef)), None)
      func = next(
        (
          p
          for p in ancestors
          if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
      )
      arg = next((p.arg for p in ancestors if isinstance(p, ast.arg)), 'return')
      found.append(
        (stub, cls.name if cls else '', func.name if func else '', arg)
      )
  assert sorted(found) == sorted(allowed), f'{root}: unexpected Any: {found}'


def check_docs(cwd: Path, package: Path, ty_cmd: list[str], ruff: str) -> None:
  required = {
    ('_structs.pyi', 'MjModel'): True,
    ('_structs.pyi', 'MjData'): True,
    ('_structs.pyi', 'from_xml_string'): False,
    ('_specs.pyi', 'MjSpec'): True,
    ('_specs.pyi', 'compile'): False,
    ('_functions.pyi', 'mj_step'): True,
    ('_functions.pyi', 'mj_forward'): True,
    ('_functions.pyi', 'mj_jac'): True,
    ('_callbacks.pyi', 'set_mjcb_control'): True,
    ('rollout.pyi', 'Rollout'): True,
    ('rollout.pyi', 'rollout'): True,
    ('rendering/classic/renderer.pyi', 'Renderer'): False,
    ('viewer.pyi', 'Handle'): False,
    ('viewer.pyi', 'launch_passive'): False,
  }
  documented = set()
  exemplified = set()
  parser = doctest.DocTestParser()
  for stub in STUBS:
    path = package / stub
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
      if not isinstance(
        node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
      ):
        continue
      doc = ast.get_docstring(node)
      if not doc:
        continue
      name = getattr(node, 'name', 'module')
      key = (stub, name)
      label = f'{stub}:{name}'
      test = parser.get_doctest(
        doc, {}, label, str(path), getattr(node, 'lineno', 0)
      )
      if key in required:
        assert len(doc.split()) >= 6, f'{label}: expected useful documentation'
        documented.add(key)
        if test.examples:
          exemplified.add(key)
      if key in {
        ('viewer.pyi', 'Handle'),
        ('rendering/classic/renderer.pyi', 'Renderer'),
      }:
        assert '```python' in doc, f'{label}: missing plain-code example'
      for example in test.examples:
        assert doctest.SKIP not in example.options, (
          f'{label}: SKIP is forbidden'
        )
      assert not (
        stub
        in {
          'viewer.pyi',
          'gl_context.pyi',
          'rendering/classic/gl_context.pyi',
          'rendering/classic/renderer.pyi',
        }
        and test.examples
      ), f'{label}: GUI/OpenGL examples must not run as doctests'
      if not test.examples:
        continue
      source = cwd / f'{stub.replace("/", "_").removesuffix(".pyi")}_{name}.py'
      source.write_text(''.join(example.source for example in test.examples))
      command([*ty_cmd, str(source)], cwd)
      command(
        [
          ruff,
          'check',
          '--config',
          str(PYTHON / 'pyproject.toml'),
          '--ignore',
          'I',
          str(source),
        ],
        cwd,
      )
      output = []
      previous = Path.cwd()
      try:
        os.chdir(cwd)
        result = doctest.DocTestRunner().run(test, out=output.append)
      finally:
        os.chdir(previous)
      assert not result.failed, f'{label}: {"".join(output)}'
  assert required.keys() <= documented, (
    f'Missing core stub docs: {sorted(required.keys() - documented)}'
  )
  expected_examples = {
    key for key, needs_example in required.items() if needs_example
  }
  assert expected_examples <= exemplified, (
    f'Missing core stub examples: {sorted(expected_examples - exemplified)}'
  )


def check_optional_sources(cwd: Path, package: Path) -> None:
  codegen = PYTHON / 'mujoco/codegen'
  command(
    [sys.executable, '-I', str(codegen / 'generate_filament_stubs.py')], cwd
  )
  spec = importlib.util.spec_from_file_location(
    'generate_studio_stubs', codegen / 'generate_studio_stubs.py'
  )
  assert spec and spec.loader
  generator = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(generator)
  for name in ('dear_imgui', 'implot'):
    stub = f'experimental/{name}/{name}.pyi'
    expected = ast.dump(ast.parse(generator.generate(name)))
    for root in (PYTHON / 'mujoco', package):
      assert ast.dump(ast.parse((root / stub).read_text())) == expected, (
        f'{root / stub}: generated UI API differs from C++ bindings'
      )


def check_typing(cwd: Path, package: Path) -> None:
  ENV.update(XDG_CONFIG_HOME=str(cwd), APPDATA=str(cwd), LOCALAPPDATA=str(cwd))
  config = cwd / 'ty.toml'
  config.write_text('')
  ty = shutil.which('ty')
  ruff = shutil.which('ruff')
  assert ty and ruff, 'ty and ruff must be on PATH'
  stubs = [package / stub for stub in STUBS]
  local_stubs = [PYTHON / 'mujoco' / stub for stub in STUBS]
  generators = [
    PYTHON / 'mujoco/codegen' / name
    for name in (
      'generate_stubs.py',
      'generate_filament_stubs.py',
      'generate_studio_stubs.py',
    )
  ]
  assert all(p.is_file() for p in generators + local_stubs), (
    'Checked-in generators and stubs are required'
  )
  check_core_annotations(PYTHON / 'mujoco')
  check_optional_sources(cwd, package)
  consumers = []
  for name in ('consumer.py', 'filament_consumer.py', 'studio_consumer.py'):
    source = Path(__file__).with_name(name)
    target = cwd / name
    shutil.copyfile(source, target)
    consumers.append((source, target))
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
      *(str(p) for p in generators),
      str(Path(__file__)),
      *(str(source) for source, _ in consumers),
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
      *(str(p) for p in generators),
      str(Path(__file__)),
      *(str(source) for source, _ in consumers),
      *(str(p) for p in local_stubs),
    ],
    cwd,
  )
  command([*ty_cmd, *(str(p) for p in stubs)], cwd)
  for _, consumer in consumers:
    command([*ty_cmd, str(consumer)], cwd)
  check_docs(cwd, package, ty_cmd, ruff)
  command(
    [
      *ty_cmd,
      '--python-version',
      '3.11',
      str(Path(__file__)),
      *(str(p) for p in generators),
    ],
    cwd,
  )

  prefix = (
    'import mujoco\nimport numpy as np\n'
    'from mujoco import _simulate, mjrf\n'
    'from mujoco.experimental.dear_imgui import dear_imgui as imgui\n'
    'from mujoco.experimental.implot import implot\n'
    'from mujoco.experimental.studio import sim, ux\n'
    'from mujoco.experimental.studio.web import headless_ui\n\n'
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
    'attach_site': (
      'spec.attach(mujoco.MjSpec(), site=42)',
      'invalid-argument-type',
    ),
    'attach_frame': (
      'spec.attach(mujoco.MjSpec(), frame=42)',
      'invalid-argument-type',
    ),
    'copy_back': ('spec.copy_back(data)', 'invalid-argument-type'),
    'encode': (
      "spec.encode('model.mjb', model=data)",
      'invalid-argument-type',
    ),
    'log_body': (
      'message: str = mujoco.MjLogMessage().body',
      'invalid-assignment',
    ),
    'wrap_target': (
      "target: mujoco.MjsSite = spec.add_tendon().wrap_site('site').target",
      'invalid-assignment',
    ),
    'grouped_bind': (
      'model.bind([spec.worldbody]).not_a_group_field',
      'unresolved-attribute',
    ),
    'simulate_images': (
      '_simulate.Simulate(mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None, False, None).set_images([(mujoco.MjrRect(0, 0, 1, 1), "bad")])',
      'invalid-argument-type',
    ),
    'filament_config': (
      "mjrf.ContextConfig(graphics_api='bad')",
      'invalid-argument-type',
    ),
    'studio_pause': (
      "sim.StepControl().set_pause_state('bad')",
      'invalid-argument-type',
    ),
    'studio_theme': (
      "ux.load_theme(def_theme='bad')",
      'invalid-argument-type',
    ),
    'studio_web': (
      "headless_ui.HeadlessUi('Studio').upload_image(0, 'bad', 1, 1, 4)",
      'invalid-argument-type',
    ),
    'imgui_begin': ('imgui.Begin(42)', 'invalid-argument-type'),
    'implot_begin': ('implot.BeginPlot(42)', 'invalid-argument-type'),
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
  message = mujoco.MjLogMessage()
  assert message.body is None and message.file is None and message.func is None
  compiled.geom('ball').size[0] = 0.2
  spec.copy_back(compiled)
  geom = spec.geom('ball')
  assert geom is not None and abs(geom.size[0] - 0.2) < 1e-6
  assert len(compiled.bind([body]).pos) == 3
  bound = mujoco.MjData(compiled)
  assert len(bound.bind([body]).xpos) == 3
  joint = spec.joint('joint')
  assert joint is not None and bound.bind([joint]).cdof[0].shape == (6,)
  parent = mujoco.MjSpec()
  parent.worldbody.add_site(name='anchor')
  child = mujoco.MjSpec()
  child.worldbody.add_body(name='attached').add_geom(size=[0.1, 0, 0], mass=1)
  assert isinstance(parent.attach(child, site='anchor'), mujoco.MjsFrame)
  assert parent.compile().nbody == 2


def main() -> None:
  with tempfile.TemporaryDirectory() as directory:
    cwd = Path(directory)
    package = installed_package(cwd)
    check_core_annotations(package)
    check_native_api(package)
    smoke()
    check_typing(cwd, package)
  print(
    'Installed wheel, runtime, native API, docs, Ruff, and ty checks passed.'
  )


if __name__ == '__main__':
  main()
