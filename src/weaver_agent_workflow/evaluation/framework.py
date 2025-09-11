from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class KernelRecord:
    name: str
    code: str
    metrics: Dict[str, Any]


class EvaluationFramework:
    def __init__(self):
        from ..fsr.framework import CompilationVerifier, FunctionValidator, PerformanceProfiler
        self.compiler = CompilationVerifier()
        self.validator = FunctionValidator()
        self.profiler = PerformanceProfiler()

    def evaluate_correctness(self, generated_kernels: List[KernelRecord], test_suite: Any) -> Dict[str, Any]:
        results = {}
        for kernel in generated_kernels:
            compile_result = self.compiler.verify(kernel.code)
            if not compile_result.success:
                results[kernel.name] = {
                    "passed": False,
                    "errors": compile_result.errors,
                    "stage": "compilation"
                }
                continue
                
            validation_result = self.validator.validate(kernel.code, test_suite)
            results[kernel.name] = {
                "passed": validation_result.passed,
                "errors": validation_result.mismatches,
                "stage": "validation" if validation_result.passed else "runtime"
            }
        return results

    def evaluate_performance(self, generated_kernels: List[KernelRecord], test_suite: Any) -> Dict[str, Any]:
        results = {}
        for kernel in generated_kernels:
            try:
                profile_result = self.profiler.profile(kernel.code, test_suite)
                results[kernel.name] = {
                    "time_ms": profile_result.avg_ms,
                    "runs": profile_result.runs,
                    "valid": profile_result.avg_ms < float("inf")
                }
            except Exception as e:
                results[kernel.name] = {
                    "time_ms": float("inf"),
                    "runs": 0,
                    "valid": False,
                    "error": str(e)
                }
        return results

    def calculate_speedups(self, performance_results: Dict[str, Any], baseline_time: float) -> Dict[str, Any]:
        speedup_results = {}
        for name, perf in performance_results.items():
            if perf["valid"] and perf["time_ms"] > 0:
                speedup = baseline_time / perf["time_ms"]
                speedup_results[name] = {
                    **perf,
                    "speedup": speedup,
                    "baseline_time_ms": baseline_time
                }
            else:
                speedup_results[name] = {
                    **perf,
                    "speedup": 0.0,
                    "baseline_time_ms": baseline_time
                }
        return speedup_results

    def generate_report(self, correctness_results: Dict[str, Any], 
                       performance_results: Dict[str, Any]) -> str:
        total_kernels = len(correctness_results)
        passed_kernels = sum(1 for r in correctness_results.values() if r["passed"])
        valid_performance = sum(1 for r in performance_results.values() if r.get("valid", False))
        
        best_kernel = None
        best_time = float("inf")
        for name, perf in performance_results.items():
            if perf.get("valid", False) and perf["time_ms"] < best_time:
                best_time = perf["time_ms"]
                best_kernel = name
        
        report_lines = [
            f"Evaluation Summary:",
            f"Total kernels evaluated: {total_kernels}",
            f"Kernels passed correctness: {passed_kernels}/{total_kernels}",
            f"Kernels with valid performance: {valid_performance}/{total_kernels}",
        ]
        
        if best_kernel:
            report_lines.extend([
                f"Best performing kernel: {best_kernel}",
                f"Best execution time: {best_time:.3f} ms"
            ])
            
        if "speedup" in next(iter(performance_results.values()), {}):
            speedups = [r.get("speedup", 0) for r in performance_results.values() 
                       if r.get("valid", False)]
            if speedups:
                max_speedup = max(speedups)
                avg_speedup = sum(speedups) / len(speedups)
                report_lines.extend([
                    f"Maximum speedup achieved: {max_speedup:.2f}x",
                    f"Average speedup: {avg_speedup:.2f}x"
                ])
        
        return "\n".join(report_lines)
