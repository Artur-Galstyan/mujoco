"""Verify installed typed-mujoco against the installed official MuJoCo wheel."""

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

ROOT = Path(__file__).resolve().parents[2]
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
  '_render_filament.pyi',
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


def installed_stubs() -> Path:
  import mujoco

  upstream = importlib.metadata.distribution('mujoco')
  companion = importlib.metadata.distribution('typed-mujoco')
  assert companion.version == '3.15.0.0', companion.version
  assert any(
    req.replace(' ', '') == 'mujoco>=3.15.0' for req in companion.requires or []
  ), 'Companion must require MuJoCo 3.15.0 or newer'
  assert upstream.files is not None and companion.files is not None
  upstream_files = {p.as_posix() for p in upstream.files}
  owned = {p.as_posix() for p in companion.files}
  stub_files = {p for p in owned if p.startswith('mujoco-stubs/')}
  expected = {'mujoco-stubs/py.typed', *(f'mujoco-stubs/{p}' for p in STUBS)}
  assert len(STUBS) == 29 and stub_files == expected, (
    f'missing: {sorted(expected - stub_files)}; extra: {sorted(stub_files - expected)}'
  )
  assert not (owned & upstream_files), 'Overlapping distribution records'
  other_files = owned - stub_files
  assert all(
    p.startswith('typed_mujoco-3.15.0.0.dist-info/') for p in other_files
  ), f'Companion contains non-stub files: {sorted(other_files)}'
  assert 'mujoco/__init__.py' in upstream_files
  package = Path(str(companion.locate_file('mujoco-stubs'))).resolve()
  runtime = Path(str(upstream.locate_file('mujoco/__init__.py'))).resolve()
  assert package.is_dir() and runtime.is_file()
  assert package != (ROOT / 'typed-mujoco/mujoco-stubs').resolve()
  assert runtime != (ROOT / 'python/mujoco/__init__.py').resolve()
  assert (package / 'py.typed').read_text().strip() == 'partial'
  assert all((package / name).is_file() for name in STUBS)
  assert not (package / 'mjrf.pyi').exists()
  spec = importlib.util.find_spec('mujoco')
  assert spec and spec.origin and Path(spec.origin).resolve() == runtime
  assert Path(mujoco.__file__).resolve() == runtime
  return package


def declared(nodes: list[ast.stmt]) -> set[str]:
  names = {
    node.name
    for node in nodes
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
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
  return {name for name in names if not name.startswith('_')}


def check_class(label: str, node: ast.ClassDef, runtime: type) -> None:
  actual = {name for name in vars(runtime) if not name.startswith('_')}
  if label == '_specs.MjVisual':
    actual.discard('global')  # A Python keyword cannot be a declared field.
  promised = declared(node.body)
  assert actual == promised, (
    f'{label}: missing {sorted(actual - promised)}, extra {sorted(promised - actual)}'
  )
  setters = {
    member.name
    for member in node.body
    if isinstance(member, ast.FunctionDef)
    and any(
      isinstance(d, ast.Attribute) and d.attr == 'setter'
      for d in member.decorator_list
    )
  }
  for member in node.body:
    if isinstance(member, ast.ClassDef) and not member.name.startswith('_'):
      child = getattr(runtime, member.name, None)
      assert isinstance(child, type), f'{label}.{member.name}: missing class'
      check_class(f'{label}.{member.name}', member, child)
    if isinstance(member, ast.FunctionDef) and any(
      isinstance(d, ast.Name) and d.id == 'property'
      for d in member.decorator_list
    ):
      prop = vars(runtime).get(member.name)
      assert isinstance(prop, property), f'{label}.{member.name}: no property'
      assert (member.name in setters) == (prop.fset is not None), (
        f'{label}.{member.name}: wrong setter flag'
      )


def check_native_api(package: Path) -> None:
  import mujoco

  native_exports = set()
  for name in NATIVE:
    module = importlib.import_module(f'mujoco.{name}')
    tree = ast.parse((package / f'{name}.pyi').read_text())
    actual = {key for key in vars(module) if not key.startswith('_')}
    promised = declared(tree.body)
    assert actual == promised, (
      f'{name}: missing {sorted(actual - promised)}, extra {sorted(promised - actual)}'
    )
    for node in tree.body:
      if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
        check_class(f'{name}.{node.name}', node, getattr(module, node.name))
    if name not in {'_simulate', '_rollout'}:
      native_exports.update(actual)

  root = ast.parse((package / '__init__.pyi').read_text())
  exports = {
    alias.asname or alias.name
    for node in root.body
    if isinstance(node, ast.ImportFrom)
    and node.module
    and node.module in {f'mujoco.{name}' for name in NATIVE}
    for alias in node.names
    if not alias.name.startswith('_')
  }
  missing = native_exports & vars(mujoco).keys() - exports - declared(root.body)
  extra = exports - vars(mujoco).keys()
  assert not missing and not extra, (
    f'mujoco: missing root exports {sorted(missing)}, extra {sorted(extra)}'
  )


def run(
  args: list[str], cwd: Path, env: dict[str, str], status: int = 0
) -> str:
  result = subprocess.run(
    args, cwd=cwd, env=env, text=True, capture_output=True, timeout=120
  )
  output = result.stdout + result.stderr
  assert result.returncode == status, (
    f'{args!r}: expected {status}, got {result.returncode}:\n{output}'
  )
  return output


def check_docs(
  package: Path, cwd: Path, env: dict[str, str], ty_cmd: list[str], ruff: str
) -> None:
  required = {
    ('_structs.pyi', 'MjModel'),
    ('_structs.pyi', 'MjData'),
    ('_specs.pyi', 'MjSpec'),
    ('_functions.pyi', 'mj_step'),
    ('_functions.pyi', 'mj_forward'),
    ('_functions.pyi', 'mj_jac'),
    ('_callbacks.pyi', 'set_mjcb_control'),
    ('rollout.pyi', 'Rollout'),
    ('rollout.pyi', 'rollout'),
  }
  seen = set()
  parser = doctest.DocTestParser()
  for stub in (
    '_structs.pyi',
    '_specs.pyi',
    '_functions.pyi',
    '_callbacks.pyi',
    'rollout.pyi',
  ):
    path = package / stub
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
      if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
        continue
      doc = ast.get_docstring(node)
      if not doc:
        continue
      name = getattr(node, 'name', 'module')
      test = parser.get_doctest(doc, {}, f'{stub}:{name}', str(path), 0)
      if (stub, name) in required and test.examples:
        seen.add((stub, name))
      if not test.examples:
        continue

      assert all(
        doctest.SKIP not in example.options for example in test.examples
      )
      source = cwd / f'doc_{stub.removesuffix(".pyi")}_{name}.py'
      source.write_text(''.join(example.source for example in test.examples))
      run([*ty_cmd, str(source)], cwd, env)
      run(
        [
          ruff,
          'check',
          '--config',
          str(ROOT / 'typed-mujoco/pyproject.toml'),
          '--ignore',
          'I',
          str(source),
        ],
        cwd,
        env,
      )
      source.unlink()
      output = []
      result = doctest.DocTestRunner().run(test, out=output.append)
      assert not result.failed, f'{stub}:{name}: {"".join(output)}'
  assert seen == required, (
    f'Missing documented examples: {sorted(required - seen)}'
  )


def smoke() -> None:
  import mujoco

  xml = (
    '<mujoco><worldbody><body name="ball"><freejoint/>'
    '<geom size=".1"/></body></worldbody></mujoco>'
  )
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  before = mujoco.get_mjcb_control()
  calls = []
  try:
    mujoco.set_mjcb_control(lambda m, d: calls.append(d.time))
    mujoco.mj_step(model, data)
  finally:
    mujoco.set_mjcb_control(before)
  assert data.time > 0 and calls
  spec = mujoco.MjSpec.from_string(xml)
  assert spec.compile().nbody == model.nbody
  archive = io.BytesIO()
  mujoco.to_zip(spec, archive)
  archive.seek(0)
  assert mujoco.from_zip(archive).compile().nbody == model.nbody


def check_typing(cwd: Path, package: Path) -> None:
  assert not cwd.resolve().is_relative_to(ROOT / 'typed-mujoco')
  env = {
    key: val
    for key, val in os.environ.items()
    if key.upper()
    not in {
      'PYTHONPATH',
      'PYTHONHOME',
      'PYTHONSTARTUP',
      'XDG_CONFIG_HOME',
      'APPDATA',
      'LOCALAPPDATA',
    }
    and not key.upper().startswith('TY_')
  }
  env.update(
    HOME=str(cwd),
    XDG_CONFIG_HOME=str(cwd),
    APPDATA=str(cwd),
    LOCALAPPDATA=str(cwd),
    PYTHONNOUSERSITE='1',
    PYTHONDONTWRITEBYTECODE='1',
  )
  config = cwd / 'ty.toml'
  config.write_text('')
  ty = shutil.which('ty')
  ruff = shutil.which('ruff')
  assert ty and ruff, 'ty and ruff must be on PATH'
  cmd = [
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
  for name in ('consumer.py', 'filament_consumer.py', 'studio_consumer.py'):
    shutil.copyfile(Path(__file__).with_name(name), cwd / name)
  run(
    [
      ruff,
      'check',
      '--config',
      str(ROOT / 'typed-mujoco/pyproject.toml'),
      str(Path(__file__)),
      *(str(p) for p in Path(__file__).parent.glob('*consumer.py')),
    ],
    cwd,
    env,
  )
  run(
    [
      ruff,
      'format',
      '--check',
      '--config',
      str(ROOT / 'typed-mujoco/pyproject.toml'),
      str(Path(__file__)),
      *(str(p) for p in Path(__file__).parent.glob('*consumer.py')),
    ],
    cwd,
    env,
  )
  run([*cmd, str(Path(__file__))], cwd, env)
  for name in ('consumer.py', 'filament_consumer.py', 'studio_consumer.py'):
    run([*cmd, str(cwd / name)], cwd, env)
  run([*cmd, *(str(package / stub) for stub in STUBS)], cwd, env)
  check_docs(package, cwd, env, cmd, ruff)

  upstream = importlib.metadata.distribution('mujoco')
  minimize = Path(str(upstream.locate_file('mujoco/minimize.py'))).resolve()
  assert upstream.files is not None
  upstream_files = {p.as_posix() for p in upstream.files}
  assert 'mujoco/minimize.py' in upstream_files
  assert 'mujoco/minimize.pyi' not in upstream_files
  assert minimize.is_file() and not (package / 'minimize.pyi').exists()
  origin = run(
    [
      sys.executable,
      '-I',
      '-c',
      'import mujoco.minimize as m; assert m.Status.MAX_ITER in m.Status; print(m.__file__)',
    ],
    cwd,
    env,
  ).strip()
  assert Path(origin).resolve() == minimize
  fallback = cwd / 'fallback.py'
  fallback.write_text('import mujoco.minimize as minimize\n')
  output = run([*cmd, str(fallback)], cwd, env, status=1)
  found = re.findall(r'(?m)^.*?:(\d+):\d+: (error|warning)\[([^]]+)\]', output)
  assert found == [('1', 'error', 'unresolved-import')], output
  assert 'Cannot resolve imported module `mujoco.minimize`' in output
  fallback.unlink()
  print(
    'Known limitation: ty 0.0.85 cannot resolve upstream-only modules; MJX fallback is not covered'
  )
  negatives = {
    'bad_step': ("mujoco.mj_step(model, 'not data')", 'invalid-argument-type'),
    'bad_nstep': (
      "mujoco.mj_step(model, data, nstep='no')",
      'invalid-argument-type',
    ),
    'bad_array': (
      'mujoco.mj_jac(model, data, [], np.zeros((3, model.nv)), np.zeros(3), 0)',
      'invalid-argument-type',
    ),
    'bad_field': ("data.time = 'no'", 'invalid-assignment'),
    'bad_spec': ("spec.worldbody.add_body(pos='bad')", 'invalid-argument-type'),
    'bad_callback': (
      'mujoco.set_mjcb_control(lambda: None)',
      'invalid-argument-type',
    ),
    'bad_filament': (
      "mjrf.ContextConfig(graphics_api='bad')",
      'invalid-argument-type',
    ),
    'bad_studio': ("ux.load_theme(def_theme='bad')", 'invalid-argument-type'),
    'old_spec': ('spec.copy_back(model)', 'unresolved-attribute'),
    'old_actuator': ('mujoco.mjtActuator', 'unresolved-attribute'),
    'old_enum': (
      'mujoco.mjtCtrlInput.mjINPUT_PRESSURE',
      'unresolved-attribute',
    ),
  }
  prefix = (
    'import numpy as np\nimport mujoco\n'
    'from mujoco import _render_filament as mjrf\n'
    'from mujoco.experimental.studio import ux\n'
    'model = mujoco.MjModel.from_xml_string("<mujoco/>")\n'
    'data = mujoco.MjData(model)\nspec = mujoco.MjSpec()\n'
  )
  line = len(prefix.splitlines()) + 1
  for name, (code, rule) in negatives.items():
    source = cwd / f'{name}.py'
    source.write_text(prefix + code + '\n')
    output = run([*cmd, str(source)], cwd, env, status=1)
    found = re.findall(
      r'(?m)^.*?:(\d+):\d+: (error|warning)\[([^]]+)\]', output
    )
    assert found == [(str(line), 'error', rule)], f'{name}: {output}'
    source.unlink()


def main() -> None:
  package = installed_stubs()
  check_native_api(package)
  smoke()
  with tempfile.TemporaryDirectory() as directory:
    check_typing(Path(directory), package)
  print('Installed companion, runtime, native API, docs, ty and Ruff passed')


if __name__ == '__main__':
  main()
