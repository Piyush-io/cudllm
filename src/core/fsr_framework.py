import os
from .llm_interface import LLMClient
from .prompt_manager import PromptManager
from ..execution_checker.compilation_checker import CompilationVerifier
from ..execution_checker.functional_validator import FunctionValidator
from ..execution_checker.performance_profiler import PerformanceProfiler
from typing import List, Tuple
import os
from ..schemas.llm_response import (
    CompileResult,
    ValidationResult,
    ProfileResult,
    CandidateResult,
    FSRSearchResult,
)


class FSR_Framework:

    def __init__(self, max_depth: int = 5, candidates_per_round: int = 5):
        self.llm = LLMClient()
        self.prompts = PromptManager()
        self.max_depth = max_depth
        self.N = candidates_per_round
        self.compilation_verifier = CompilationVerifier()
        self.function_validator = FunctionValidator()
        self.performance_profiler = PerformanceProfiler()

    def generate_kernels(self, prompt: str, num_kernels: int) -> List[str]:
        return self.llm.generate_kernels(prompt, num_kernels)

    def fsr_search(self, task_desc: str, host_code: str, hw_spec: dict) -> FSRSearchResult:
        current_prompt = self.prompts.create_initial_prompt(task_desc, host_code, hw_spec, self.N)
        if "arch" in hw_spec:
            self.compilation_verifier.arch = hw_spec["arch"]
        best_kernel = None
        best_time = float("inf")
        error_history: List[str] = []
        all_candidates: List[CandidateResult] = []
        iterations = 0

        for _ in range(self.max_depth):
            candidates = self.generate_kernels(current_prompt, self.N)
            iterations += 1
            compiled: List[Tuple[str, str]] = []
            compile_errors: List[str] = []
            for cand in candidates:
                ok, bin_path, err = self.compilation_verifier.verify(cand)
                all_candidates.append(
                    CandidateResult(
                        kernel_src=cand,
                        compile=CompileResult(ok=ok, binary_path=bin_path if ok else None, stderr=None if ok else err),
                    )
                )
                if ok:
                    compiled.append((cand, bin_path))
                else:
                    compile_errors.append(err)

            if not compiled:
                error_history.extend(compile_errors)
                current_prompt = self.prompts.refine_prompt_for_errors(current_prompt, compile_errors, error_history)
                continue

            validated: List[Tuple[str, str]] = []
            validation_errors: List[str] = []
            for cand, bin_path in compiled:
                ok, out = self.function_validator.validate(bin_path)
                for i in range(len(all_candidates) - 1, -1, -1):
                    if all_candidates[i].kernel_src == cand and all_candidates[i].validate is None:
                        all_candidates[i].validatation = ValidationResult(ok=ok, output=out)
                        break
                if ok:
                    validated.append((cand, bin_path))
                else:
                    validation_errors.append(out)

            if not validated:
                error_history.extend(validation_errors)
                current_prompt = self.prompts.refine_prompt_for_errors(current_prompt, validation_errors, error_history)
                continue

            fastest_kernel = None
            fastest_time = float("inf")
            best_out = ""
            for cand, bin_path in validated:
                ok, time_ms, out = self.performance_profiler.profile(bin_path)
                for i in range(len(all_candidates) - 1, -1, -1):
                    if all_candidates[i].kernel_src == cand:
                        all_candidates[i].profile = ProfileResult(ok=ok, time_ms=time_ms, output=out)
                        break
                if ok and time_ms < fastest_time:
                    fastest_time = time_ms
                    fastest_kernel = cand
                    best_out = out

            if fastest_kernel is None:
                error_history.append("profiling failed")
                current_prompt = self.prompts.refine_prompt_for_errors(current_prompt, ["profiling failed"], error_history)
                continue

            if fastest_time < best_time:
                best_time = fastest_time
                best_kernel = fastest_kernel

            current_prompt = self.prompts.refine_prompt_for_performance(current_prompt, best_out)

        return FSRSearchResult(
            best_kernel=best_kernel,
            best_time_ms=best_time if best_time != float("inf") else -1.0,
            iterations=iterations,
            candidates=all_candidates,
        )
