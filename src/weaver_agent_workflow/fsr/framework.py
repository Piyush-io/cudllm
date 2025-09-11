from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any, Tuple
import tempfile
import subprocess
import os
import shutil
from pathlib import Path
import numpy as np
import re


@dataclass
class CandidateKernel:
    code: str
    name: str
    metadata: dict


@dataclass
class VerificationResult:
    success: bool
    errors: Optional[List[str]] = None


@dataclass
class ValidationResult:
    passed: bool
    mismatches: Optional[List[str]] = None


@dataclass
class ProfileResult:
    avg_ms: float
    runs: int


class CompilationVerifier:
    def _nvcc_path(self) -> Optional[str]:
        return shutil.which("nvcc")

    def verify(self, cuda_code: str, device_info: Optional[dict] = None) -> VerificationResult:
        nvcc = self._nvcc_path()
        if not nvcc:
            return VerificationResult(success=False, errors=["nvcc not found in PATH"])

        try:
            with tempfile.TemporaryDirectory() as td:
                cu_path = Path(td) / "kernel.cu"
                out_path = Path(td) / "kernel.ptx"
                cu_path.write_text(cuda_code)
                cmd = [nvcc, "-ptx", str(cu_path), "-o", str(out_path), "-O3", "--use_fast_math"]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode != 0:
                    return VerificationResult(success=False, errors=[proc.stderr.strip() or proc.stdout.strip()])
                return VerificationResult(success=True, errors=None)
        except Exception as e:
            return VerificationResult(success=False, errors=[f"verify exception: {e}"])


class FunctionValidator:
    def __init__(self, reference_implementation: str = "identity"):
        self.reference_implementation = reference_implementation
    
    def _compile_and_load(self, cuda_code: str, kernel_name: str):
        try:
            import cupy as cp  # type: ignore
        except ImportError:
            raise ImportError("CuPy not installed. Install appropriate CuPy package for your CUDA version.")
        
        try:
            module = cp.RawModule(code=cuda_code, options=("-O3", "--use_fast_math"))
            func = module.get_function(kernel_name)
            return cp, func
        except cp.cuda.compiler.CompileException as e:
            raise RuntimeError(f"CUDA compilation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CUDA kernel '{kernel_name}': {e}")

    def _compute_expected_output(self, x: np.ndarray) -> np.ndarray:
        """Compute expected output based on reference implementation."""
        if self.reference_implementation == "identity":
            return x.copy()
        elif self.reference_implementation == "vector_add":
            return x + x
        elif self.reference_implementation == "scale":
            return x * 2.0
        elif self.reference_implementation == "square":
            return x * x
        else:
            return x.copy()

    def validate(self, cuda_code: str, test_cases: Any) -> ValidationResult:
        try:
            if not isinstance(test_cases, dict):
                return ValidationResult(passed=False, mismatches=["test_cases must be a dictionary"])
                
            xs: List[np.ndarray] = list(test_cases.get("inputs", []))
            if not xs:
                return ValidationResult(passed=False, mismatches=["no test inputs provided"])

            kernel_name = _extract_kernel_name(cuda_code)
            if not kernel_name or kernel_name == "kernel_0":
                return ValidationResult(passed=False, mismatches=[f"could not extract kernel name from code"])
                
            cp, func = self._compile_and_load(cuda_code, kernel_name)

            for i, x in enumerate(xs):
                if not isinstance(x, np.ndarray):
                    return ValidationResult(passed=False, mismatches=[f"input {i} is not a numpy array"])
                    
                if x.size == 0:
                    continue
                    
                n = int(x.size)
                x_gpu = cp.asarray(x)
                y_gpu = cp.empty_like(x_gpu)
                grid, block = _default_launch(n)
                
                try:
                    func((grid,), (block,), (x_gpu, y_gpu, np.int32(n)))
                    cp.cuda.runtime.deviceSynchronize()
                except Exception as e:
                    return ValidationResult(passed=False, mismatches=[f"kernel launch failed: {e}"])
                
                y = cp.asnumpy(y_gpu)
                expected = self._compute_expected_output(x)
                
                if not np.allclose(y, expected, atol=1e-5, rtol=1e-5):
                    return ValidationResult(passed=False, mismatches=[f"output mismatch for input {i}: expected {self.reference_implementation} operation"])
                    
            return ValidationResult(passed=True, mismatches=None)
            
        except ImportError as e:
            return ValidationResult(passed=False, mismatches=[str(e)])
        except RuntimeError as e:
            return ValidationResult(passed=False, mismatches=[str(e)])
        except Exception as e:
            return ValidationResult(passed=False, mismatches=[f"validation failed: {e}"])


class PerformanceProfiler:
    def _compile_and_load(self, cuda_code: str, kernel_name: str):
        try:
            import cupy as cp  # type: ignore
        except ImportError:
            raise ImportError("CuPy not installed. Install appropriate CuPy package for your CUDA version.")
        
        try:
            module = cp.RawModule(code=cuda_code, options=("-O3", "--use_fast_math"))
            func = module.get_function(kernel_name)
            return cp, func
        except cp.cuda.compiler.CompileException as e:
            raise RuntimeError(f"CUDA compilation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CUDA kernel '{kernel_name}': {e}")

    def profile(self, cuda_code: str, input_sizes: Any) -> ProfileResult:
        try:
            if not isinstance(input_sizes, dict):
                return ProfileResult(avg_ms=float("inf"), runs=0)
                
            xs: List[np.ndarray] = list(input_sizes.get("inputs", []))
            if not xs:
                return ProfileResult(avg_ms=float("inf"), runs=0)
                
            x = xs[0]
            if not isinstance(x, np.ndarray) or x.size == 0:
                return ProfileResult(avg_ms=float("inf"), runs=0)
                
            n = int(x.size)
            kernel_name = _extract_kernel_name(cuda_code)
            if not kernel_name or kernel_name == "kernel_0":
                return ProfileResult(avg_ms=float("inf"), runs=0)
                
            cp, func = self._compile_and_load(cuda_code, kernel_name)
            x_gpu = cp.asarray(x)
            y_gpu = cp.empty_like(x_gpu)
            grid, block = _default_launch(n)
            
            # warmup runs
            try:
                for _ in range(3):
                    func((grid,), (block,), (x_gpu, y_gpu, np.int32(n)))
                cp.cuda.runtime.deviceSynchronize()
            except Exception:
                return ProfileResult(avg_ms=float("inf"), runs=0)
            
            # timed runs
            try:
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                runs = 10
                start.record()
                for _ in range(runs):
                    func((grid,), (block,), (x_gpu, y_gpu, np.int32(n)))
                end.record()
                end.synchronize()
                ms = cp.cuda.get_elapsed_time(start, end) / runs
                return ProfileResult(avg_ms=float(ms), runs=runs)
            except Exception:
                return ProfileResult(avg_ms=float("inf"), runs=0)
                
        except (ImportError, RuntimeError):
            return ProfileResult(avg_ms=float("inf"), runs=0)
        except Exception:
            return ProfileResult(avg_ms=float("inf"), runs=0)


class FSRFramework:
    def __init__(self, llm_model: Any, max_depth: int = 5, candidates_per_round: int = 5, reference_implementation: str = "identity"):
        self.llm = llm_model
        self.max_depth = max_depth
        self.N = candidates_per_round
        self.reference_implementation = reference_implementation
        self.compilation_verifier = CompilationVerifier()
        self.function_validator = FunctionValidator(reference_implementation)
        self.performance_profiler = PerformanceProfiler()

    def fsr_search(self, initial_prompt: str, test_cases: Any) -> Optional[CandidateKernel]:
        best: Optional[CandidateKernel] = None
        best_time: float = float("inf")
        current_prompt = initial_prompt

        for depth in range(self.max_depth):
            candidates = self.llm.generate_kernels(current_prompt, self.N)

            compiled: List[CandidateKernel] = []
            last_compile_errors: List[str] = []
            for cand in candidates:
                v = self.compilation_verifier.verify(cand.code, device_info=None)
                if v.success:
                    compiled.append(cand)
                else:
                    last_compile_errors.extend(v.errors or [])

            if not compiled:
                current_prompt = self.llm.prompt.refine_prompt_for_errors(current_prompt, last_compile_errors, history=None)
                continue

            validated: List[CandidateKernel] = []
            last_validation_errors: List[str] = []
            for cand in compiled:
                res = self.function_validator.validate(cand.code, test_cases)
                if res.passed:
                    validated.append(cand)
                else:
                    last_validation_errors.extend(res.mismatches or [])

            if not validated:
                current_prompt = self.llm.prompt.refine_prompt_for_errors(current_prompt, last_validation_errors, history=None)
                continue

            fastest: Optional[CandidateKernel] = None
            fastest_time = float("inf")
            for cand in validated:
                prof = self.performance_profiler.profile(cand.code, input_sizes=test_cases)
                if prof.avg_ms < fastest_time:
                    fastest_time = prof.avg_ms
                    fastest = cand

            if fastest and fastest_time < best_time:
                best_time = fastest_time
                best = fastest

            current_prompt = self.llm.prompt.refine_prompt_for_performance(current_prompt, best)

        return best


def _extract_kernel_name(cuda_code: str) -> str:
    m = re.search(r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", cuda_code)
    return m.group(1) if m else "kernel_0"


def _default_launch(n: int) -> Tuple[int, int]:
    block = 256
    grid = (n + block - 1) // block
    return grid, block
