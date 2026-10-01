#!/usr/bin/env python3
"""Download a pinned PUBLIC picoagent revision on an existing Colab runtime.

Run as a small Colab exec cell. This installs pinned Python training dependencies
but never allocates compute, authenticates to GitHub, or starts training. Inspect
/content/picoagent-bootstrap/status.json after a lost response before retrying.
"""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

REPOSITORY = 'https://github.com/ItsMeSamey/picoagent.git'
DEFAULT_COMMIT = '9be3c0415822bb6a72f9b7d41b2792b60e167c2e'
PACKAGES = ('transformers==5.18.0', 'accelerate==1.15.0', 'tokenizers==0.23.2',
            'httpx[socks]==0.28.1')


def main() -> dict:
    commit = os.environ.get('PICOAGENT_SOURCE_COMMIT', DEFAULT_COMMIT)
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Source commit must be an immutable full Git SHA')
    project = Path('/content/picoagent')
    control = Path('/content/picoagent-bootstrap')
    if project.is_symlink() or control.is_symlink():
        raise ValueError('Bootstrap paths cannot be symlinks')
    control.mkdir(parents=True, exist_ok=True)
    status_file = control / 'status.json'
    lock = control / 'bootstrap.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    def status(phase, **values):
        result = {'phase':phase, 'source_commit':commit, **values}
        temporary = control / 'status.json.tmp'
        temporary.write_text(json.dumps(result, sort_keys=True) + '\n')
        os.replace(temporary, status_file)
        return result
    try:
        if (project / 'runs').exists() and any((project / 'runs').iterdir()):
            raise ValueError('Existing run output is protected; inspect rather than bootstrap over it')
        status('downloading_source')
        with (control / 'setup.log').open('a') as log:
            def run(argv):
                subprocess.run(argv, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=600,
                               env={**os.environ, 'GIT_TERMINAL_PROMPT':'0'})
            project.mkdir(parents=True, exist_ok=True)
            if not (project / '.git').exists():
                if any(project.iterdir()):
                    raise ValueError('Nonempty non-Git project directory is protected')
                run(['git','init','.'])
                run(['git','remote','add','origin',REPOSITORY])
            remote = subprocess.check_output(['git','remote','get-url','origin'], cwd=project, text=True).strip()
            if remote != REPOSITORY:
                raise ValueError('Existing checkout has a different origin')
            run(['git','fetch','--depth=1','origin',commit])
            run(['git','checkout','--detach',commit])
            actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=project, text=True).strip()
            if actual != commit:
                raise ValueError('Downloaded source revision differs from requested commit')
            status('installing_dependencies')
            import sys
            run([sys.executable,'-m','pip','install','--disable-pip-version-check',*PACKAGES])
        versions = {name:importlib.metadata.version(name) for name in
                    ('torch','transformers','accelerate','tokenizers','httpx')}
        result = status('ready', packages=versions, training_started=False,
                        repository=REPOSITORY, project=str(project))
        print('PICOAGENT_RESULT=' + json.dumps(result), flush=True)
        return result
    except BaseException as error:
        status('failed', error_type=type(error).__name__,
               returncode=getattr(error,'returncode',None), training_started=False)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
