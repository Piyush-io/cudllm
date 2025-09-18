import subprocess
from typing import Tuple


class PerformanceProfiler:
    def profile(self, binary_path: str) -> Tuple[bool, float, str]:
        proc = subprocess.run([binary_path], capture_output=True, text=True)
        out = proc.stdout
        err = proc.stderr
        combined = out if not err else f"{out}\n[stderr]\n{err}"
        time_ms = -1.0
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("TIME_MS"):
                try:
                    # Accept formats: "TIME_MS 1.23", "TIME_MS: 1.23", "TIME_MS=1.23"
                    token = (
                        stripped.replace(":", " ")
                        .replace("=", " ")
                        .split()
                    )
                    # token like ["TIME_MS", "1.23"]
                    time_ms = float(token[1]) if len(token) >= 2 else -1.0
                except Exception:
                    time_ms = -1.0
        return time_ms >= 0, time_ms, combined
