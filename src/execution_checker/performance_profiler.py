import subprocess
from typing import Tuple


class PerformanceProfiler:
    def profile(self, binary_path: str) -> Tuple[bool, float, str]:
        proc = subprocess.run([binary_path], capture_output=True, text=True)
        out = proc.stdout
        time_ms = -1.0
        for line in out.splitlines():
            if line.startswith("TIME_MS"):
                try:
                    time_ms = float(line.split()[1])
                except Exception:
                    time_ms = -1.0
        return time_ms >= 0, time_ms, out
