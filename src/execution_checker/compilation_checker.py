"""CUDA compilation verifier with temp cleanup."""

import os
import subprocess
import tempfile
import shutil
from typing import Tuple
from pathlib import Path


class CompilationVerifier:
    """Verifies CUDA kernel compilation with nvcc."""

    def __init__(self, nvcc_path: str | None = None, arch: str = "sm_80"):
        """
        Initialize compilation verifier.

        Args:
            nvcc_path: Path to nvcc compiler. Defaults to NVCC env var or 'nvcc'.
            arch: Target GPU architecture (e.g., sm_80, sm_86, sm_90).
        """
        self.nvcc_path = nvcc_path or os.environ.get("NVCC", "nvcc")
        self.arch = arch
        self.work_dir = Path(tempfile.mkdtemp(prefix="fsr_work_"))

    def verify(
        self, candidate_kernel: str, candidate_id: int = 0
    ) -> Tuple[bool, str, str]:
        """
        Compile kernel and return compilation result.

        Args:
            candidate_kernel: CUDA kernel source code as string.
            candidate_id: Unique identifier for this candidate (for file naming).

        Returns:
            Tuple of (success, binary_path, stderr):
                - success: True if compilation succeeded, False otherwise.
                - binary_path: Path to compiled binary if successful, empty string otherwise.
                - stderr: Compiler error messages if compilation failed.
        """
        src_path = self.work_dir / f"kernel_{candidate_id}.cu"
        bin_path = self.work_dir / f"kernel_{candidate_id}.out"

        # Write source to file
        src_path.write_text(candidate_kernel)

        # Compile with nvcc
        cmd = [
            self.nvcc_path,
            str(src_path),
            "-O3",
            f"-arch={self.arch}",
            "-o",
            str(bin_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        ok = proc.returncode == 0

        return ok, str(bin_path) if ok else "", proc.stderr

    def cleanup(self):
        """Remove temporary work directory and all compiled artifacts."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            # Ignore cleanup errors during deletion
            pass
