import subprocess
from typing import Tuple


class FunctionValidator:
    def validate(self, binary_path: str) -> Tuple[bool, str]:
        proc = subprocess.run([binary_path], capture_output=True, text=True)
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        combined = out if not err else f"{out}\n[stderr]\n{err}"
        ok = "OK" in out and proc.returncode == 0
        return ok, combined
