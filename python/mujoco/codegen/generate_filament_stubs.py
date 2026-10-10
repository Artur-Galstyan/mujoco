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
"""Generate mjrf reexports and check reviewed Filament signatures against bindings.

Run `python generate_filament_stubs.py` to check both stubs, or pass `--write`
to update mjrf.pyi. The native signatures stay hand-reviewed: C++ declarations
alone do not describe NumPy arrays, Python buffers, or object lifetimes.
"""

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = '''# Copyright 2026 DeepMind Technologies Limited
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
"""Convenience wrapper around the MuJoCo Filament C++/python bindings."""

'''


def binding_types(inc: str, cc: str) -> tuple[dict[str, str], set[str]]:
  variables = dict(
    re.findall(r'py::(?:enum_|class_)<[^>]+>\s+(\w+)\(m,\s*"([^"]+)"', inc)
  )
  extra = set(re.findall(r'py::class_<[^>]+>\s+cls\(m,\s*"([^"]+)"', cc))
  assert len(variables) == 33 and len(extra) == 4, (
    'New native types need review'
  )
  return variables, extra


def check_native(
  inc: str, cc: str, variables: dict[str, str], extra: set[str]
) -> None:
  tree = ast.parse((ROOT / '_render_filament.pyi').read_text())
  classes = {
    node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
  }
  expected = set(variables.values()) | extra
  assert expected == classes.keys(), (
    f'Native types differ: missing {sorted(expected - classes.keys())}, '
    f'extra {sorted(classes.keys() - expected)}'
  )
  members = {
    name: {
      node.name
      for node in cls.body
      if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    | {
      node.target.id
      for node in cls.body
      if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    for name, cls in classes.items()
  }
  bindings = re.findall(r'(\w+)\.def\(\s*"([^"]+)"', inc + cc)
  bindings += re.findall(r'DefProperty<[^>]+>\((\w+),\s*"([^"]+)"', inc)
  bindings += re.findall(r'(\w+)\.value\("([^"]+)"', inc)
  missing = [
    f'{variables[var]}.{member}'
    for var, member in bindings
    if var in variables and member not in members[variables[var]]
  ]
  for match in re.finditer(r'py::class_<[^>]+>\s+cls\(m,\s*"([^"]+)"', cc):
    end = cc.index('\n  }', match.end())
    for member in re.findall(r'cls\.def\(\s*"([^"]+)"', cc[match.end() : end]):
      if member not in members[match.group(1)]:
        missing.append(f'{match.group(1)}.{member}')
  assert not missing, f'Missing native members: {sorted(missing)}'
  constructors = list(
    re.finditer(r'(\w+)\.def\(py::init\(\[\]\(.*?\n\);', inc, flags=re.S)
  )
  assert len(constructors) == 16, 'New data constructors need review'
  for match in constructors:
    var = match.group(1)
    ctor = next(
      node
      for node in classes[variables[var]].body
      if isinstance(node, ast.FunctionDef) and node.name == '__init__'
    )
    names = re.findall(r'py::arg\("([^"]+)"', match.group())
    assert names == [arg.arg for arg in ctor.args.args[1:]], (
      f'{variables[var]} constructor arguments differ: {names}'
    )
  print(
    f'Checked {len(expected)} native types and {len(bindings)} bound members.'
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--write', action='store_true', help='Write mjrf.pyi')
  args = parser.parse_args()
  inc = (ROOT / 'render_filament_generated.cc.inc').read_text()
  cc = (ROOT / 'render_filament.cc').read_text()
  variables, extra = binding_types(inc, cc)
  check_native(inc, cc, variables, extra)
  generated = HEADER + ''.join(
    f'from mujoco._render_filament import {name} as {name}\n'
    for name in sorted(set(variables.values()) | extra, key=str.casefold)
  )
  destination = ROOT / 'mjrf.pyi'
  if args.write:
    destination.write_text(generated)
  else:
    assert destination.read_text() == generated, (
      'mjrf.pyi changed: run with --write'
    )
  print('mjrf.pyi reexports match the native bindings.')


if __name__ == '__main__':
  main()
