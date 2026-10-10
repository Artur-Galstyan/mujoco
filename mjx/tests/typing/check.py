"""Check installed MJX types outside the checkout and user configuration."""

import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
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


def main() -> None:
  with tempfile.TemporaryDirectory() as directory:
    cwd = Path(directory)
    assert not cwd.resolve().is_relative_to(ROOT), (
      'Temporary project is in the checkout'
    )
    env = {
      **ENV,
      'XDG_CONFIG_HOME': directory,
      'APPDATA': directory,
      'LOCALAPPDATA': directory,
    }
    for dist_name, module_name, filename, source in (
      (
        'mujoco',
        'mujoco',
        'mujoco/__init__.py',
        ROOT / 'python/mujoco/__init__.py',
      ),
      (
        'mujoco-mjx',
        'mujoco.mjx',
        'mujoco/mjx/__init__.py',
        ROOT / 'mjx/mujoco/mjx/__init__.py',
      ),
    ):
      distribution = importlib.metadata.distribution(dist_name)
      assert distribution.files is not None, (
        f'{dist_name} lacks installed files'
      )
      recorded = {file.as_posix() for file in distribution.files}
      assert filename in recorded, f'{dist_name} lacks {filename}'
      origin = Path(distribution.locate_file(filename)).resolve()
      spec = importlib.util.find_spec(module_name)
      assert origin.is_file() and origin != source.resolve()
      assert spec and spec.origin and Path(spec.origin).resolve() == origin, (
        f'{module_name} does not resolve to its installed distribution'
      )
      if dist_name == 'mujoco-mjx':
        marker = 'mujoco/mjx/py.typed'
        assert (
          marker in recorded
          and Path(distribution.locate_file(marker)).is_file()
        )

    config = cwd / 'ty.toml'
    config.write_text('')
    ty = Path(sys.executable).with_name('ty')
    assert ty.is_file(), f'ty not installed next to {sys.executable}'
    cmd = [
      str(ty),
      'check',
      '--config-file',
      str(config),
      '--no-respect-ignore-files',
      '--project',
      directory,
      '--python',
      sys.executable,
      '--no-progress',
      '--output-format',
      'concise',
      '--error-on-warning',
    ]
    for name in ('positive.py', 'negative.py'):
      sample = Path(__file__).with_name(name)
      target = cwd / name
      shutil.copyfile(sample, target)
      result = subprocess.run(
        [*cmd, str(target)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
      )
      output = result.stdout + result.stderr
      found = sorted(
        (int(line), severity)
        for line, severity in re.findall(
          r'(?m)^.*?:(\d+):\d+: (error|warning)\[', output
        )
      )
      expected = [
        (line, 'error')
        for line, text in enumerate(sample.read_text().splitlines(), 1)
        if '# ty-error' in text
      ]
      if (
        result.returncode != (0 if name == 'positive.py' else 1)
        or found != expected
      ):
        raise AssertionError(
          f'{name}: expected {expected}, got {found}\n{output}'
        )
      print(f'{name}: {len(found)} expected type errors, no others')


if __name__ == '__main__':
  main()
