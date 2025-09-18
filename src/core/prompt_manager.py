from typing import List, Dict
from ..schemas.knowledge import StructuredUserIntent
from .hierarchical_rag import HierarchicalRAG


class PromptManager:
    def __init__(self) -> None:
        self.rag = HierarchicalRAG()

    def create_initial_prompt(self, task_desc: str, host_code: str, hw_spec: Dict[str, str], n: int) -> str:
        arch = hw_spec.get("arch", "sm_80")
        kb_text = ""
        kb_summary = ""
        try:
            intent = StructuredUserIntent(task=task_desc, hardware_arch=arch)
            kb = self.rag.retrieve_knowledge(intent)
            kb_summary = self.rag.get_retrieval_summary(kb)
            snippets: List[str] = []
            if kb.concepts:
                snippets += kb.concepts.snippets[:2]
            if kb.patterns:
                snippets += kb.patterns.snippets[:2]
            if kb.hardware:
                snippets += kb.hardware.snippets[:2]
            if kb.api:
                snippets += kb.api.snippets[:2]
            if kb.examples:
                snippets += kb.examples.snippets[:2]
            kb_text = "\n---\n".join(snippets[:6])
        except Exception:
            kb_text = ""
        return (
            f"Task: {task_desc}\n"
            f"Target GPU arch: {arch}.\n"
            "Write N CUDA C++ kernels (device code + minimal host main) that compile with nvcc on Linux.\n"
            "Follow this interface strictly. For each candidate, output a complete standalone program that: \n"
            "- includes <cuda_runtime.h> and <cstdio> \n"
            "- defines a __global__ kernel function \n"
            "- defines main() that allocates device buffers, initializes inputs, launches the kernel, checks correctness vs CPU reference, and prints a single line 'OK' on success and 'FAIL' on mismatch, and prints 'TIME_MS <float>' for measured kernel time in milliseconds using cudaEvent timing. \n"
            f"Produce exactly {n} candidates in JSON field 'kernels' as a list of strings. Do not add explanations.\n"
            "Host template to follow (pseudocode): allocate inputs, copy to device, warmup, measure N runs, copy back, compare, print.\n"
            f"Existing host code context (optional):\n{host_code}\n"
            + ("\nKnowledge summary: " + kb_summary + "\n" if kb_summary else "")
            + ("\nRelevant documentation snippets:\n" + kb_text if kb_text else "")
        )

    def refine_prompt_for_errors(self, original_prompt: str, errors: List[str], history: List[str]) -> str:
        joined = "\n".join(errors[-5:])
        return original_prompt + "\nPrevious attempts failed to compile or run. Address these errors:\n" + joined

    def refine_prompt_for_performance(self, original_prompt: str, best_kernel_notes: str) -> str:
        hints = (
            "Focus on performance: use coalesced memory accesses, prefer shared memory, avoid divergent branches, choose reasonable block sizes, unroll inner loops where beneficial."
        )
        return original_prompt + "\nPerformance hints:\n" + hints + "\n" + best_kernel_notes