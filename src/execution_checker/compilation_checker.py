import os
import subprocess
import tempfile
from typing import Tuple


class CompilationVerifier:
    def __init__(self, nvcc_path: str | None = None, arch: str = "sm_80"):
        self.nvcc_path = nvcc_path or os.environ.get("NVCC", "nvcc")
        self.arch = arch

    def verify(self, candidate_kernel: str) -> Tuple[bool, str, str]:
        tmpdir = tempfile.mkdtemp(prefix="fsr_cuda_")
        src_path = os.path.join(tmpdir, "kernel.cu")
        bin_path = os.path.join(tmpdir, "kernel.out")
        with open(src_path, "w") as f:
            f.write(candidate_kernel)
        cmd = [
            self.nvcc_path,
            src_path,
            "-O3",
            f"-arch={self.arch}",
            "-o",
            bin_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        ok = proc.returncode == 0
        return ok, bin_path if ok else "", proc.stderr
