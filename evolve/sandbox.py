"""Run untrusted code in a Docker container (OrbStack on the Mac).

PROTECTED: the model may not edit this file.

Every tool call the agent makes, and every unit-test scorer, goes through
``run``. The container has no network, a read-only mount of the repo, a
memory cap and a hard timeout. Python runs with -E -s (ignore env and user site) rather
than -I because -I would also drop the script directory from sys.path and
break ``from solution import ...``. If Docker is unavailable, ``run`` falls back
to a subprocess with a timeout and logs a loud warning; use that only for
dry runs on a trusted machine.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import CFG, ROOT

log = logging.getLogger(__name__)


@dataclass
class Result:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def docker_available() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0


def run_python(code: str, files: dict[str, str] | None = None,
               timeout: int | None = None) -> Result:
    """Execute ``code`` as /work/main.py with optional extra files alongside."""
    timeout = timeout or CFG.sandbox.timeout_seconds
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "main.py").write_text(code)
        for name, content in (files or {}).items():
            (work / name).write_text(content)
        if docker_available():
            cmd = [
                "docker", "run", "--rm",
                "--network", CFG.sandbox.network,
                "--memory", CFG.sandbox.memory,
                "--cpus", "2",
                "--pids-limit", "128",
                "--read-only",
                "--tmpfs", "/tmp:rw,size=64m",
                "-v", f"{work}:/work:rw",
                "-v", f"{ROOT}:/repo:ro",
                "-w", "/work",
                CFG.sandbox.image,
                "python", "-E", "-s", "main.py",
            ]
        else:
            log.warning("docker not available: running in a bare subprocess (dry-run only)")
            cmd = ["python3", "-E", "-s", str(work / "main.py")]
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=work, env={"PATH": os.environ.get("PATH", "")},
            )
            return Result(p.stdout[-20000:], p.stderr[-20000:], p.returncode)
        except subprocess.TimeoutExpired as e:
            return Result((e.stdout or b"").decode()[-20000:] if isinstance(e.stdout, bytes) else (e.stdout or ""),
                          "timeout", -1, timed_out=True)
