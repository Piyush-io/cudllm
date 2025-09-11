from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, cast
import os

from openai import OpenAI
from pydantic import BaseModel

from ..fsr.framework import CandidateKernel


@dataclass
class LLMConfig:
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.1  # Lower temperature for more deterministic CUDA code


class GeneratedCUDAKernel(BaseModel):
    code: str
    kernel_name: str
    optimizations_used: List[str]


class PromptManager:
    def create_initial_prompt(self, task_desc: str, host_code: str, hw_spec: dict) -> str:
        """Create initial prompt following the paper's format"""
        return f"""Write a CUDA kernel function on {hw_spec.get('name', 'Unknown GPU')} GPU, utilizing the functions described as below:

[Task]: {task_desc}

The output should be the content of whole .cu file containing ONE kernel function, completing the reference code below:

{host_code}

Hardware Specifications:
- Architecture: {hw_spec.get('architecture', 'Unknown')}
- Compute Capability: {hw_spec.get('compute_capability', '8.0')}
- Memory: {hw_spec.get('memory_gb', 'Unknown')} GB
- Memory Bandwidth: {hw_spec.get('memory_bandwidth_gb_s', 'Unknown')} GB/s
- CUDA Cores: {hw_spec.get('cuda_cores', 'Unknown')}
- SM Count: {hw_spec.get('sm_count', 'Unknown')}
- Max Threads per Block: {hw_spec.get('max_threads_per_block', 1024)}
- Max Shared Memory per Block: {hw_spec.get('max_shared_memory_per_block', 49152)} bytes
- Warp Size: {hw_spec.get('warp_size', 32)}

Requirements:
1. Generate high-performance CUDA kernel code optimized for the target hardware
2. Use modern CUDA optimization techniques (memory coalescing, shared memory, warp primitives)
3. Handle boundary conditions and edge cases properly
4. Include proper thread indexing and memory access patterns
5. Optimize for the specific GPU architecture and compute capability

Do not modify the test part."""

    def refine_prompt_for_errors(self, original_prompt: str, errors: List[str], history: Optional[List[str]] = None) -> str:
        """Refine prompt with compilation/runtime errors following paper's format"""
        if not errors:
            return original_prompt + "\n\nThe code failed to compile. Please fix syntax and semantic errors."

        # Determine error type and create appropriate prompt
        compilation_errors = [e for e in errors if any(keyword in e.lower() for keyword in ['error:', 'undefined', 'syntax', 'compile'])]
        runtime_errors = [e for e in errors if any(keyword in e.lower() for keyword in ['kernel launch', 'runtime', 'cuda', 'memory'])]

        if compilation_errors:
            error_snippet = "\n".join(compilation_errors[:3])
            return f"""Modify the code with the execution error result.
The output should be the content of whole .cu file containing ONE kernel function.
Do not modify the test part.

The execution output is:
{error_snippet}

Original task requirements:
{original_prompt}"""

        elif runtime_errors:
            return f"""The code failed to launch the kernel. Modify the code with the device information.
The output should be the content of whole .cu file containing ONE kernel function.
Do not modify the test part.

Runtime errors encountered:
{chr(10).join(runtime_errors[:2])}

Original task requirements:
{original_prompt}"""

        else:
            return f"""The result is not the same with the reference output. Modify the code.
The output should be the content of whole .cu file containing ONE kernel function.
Do not modify the test part.

Functional errors:
{chr(10).join(errors[:2])}

Original task requirements:
{original_prompt}"""

    def refine_prompt_for_performance(self, original_prompt: str, best_kernel: Optional[CandidateKernel]) -> str:
        """Refine prompt for performance optimization following paper's format"""
        performance_hint = ""
        if best_kernel and best_kernel.metadata:
            exec_time = best_kernel.metadata.get('execution_time', 'unknown')
            performance_hint = f"Best so far: {best_kernel.name} with execution time {exec_time}ms."

        return f"""Optimize the kernel function for less execution time on the target GPU.
The output should be the content of whole .cu file containing ONE kernel function.
Do not modify the test part.

{performance_hint}

Focus on these optimization strategies:
1. Memory coalescing - ensure contiguous memory access patterns
2. Shared memory utilization - reduce global memory accesses
3. Warp-level primitives - use __shfl_*, __ballot_sync for efficient communication
4. Occupancy optimization - balance threads per block and register usage
5. Loop unrolling - reduce loop overhead with #pragma unroll
6. Memory access optimization - minimize scattered reads/writes
7. Avoid warp divergence - minimize branching within warps

Original task requirements:
{original_prompt}"""


class LLMInterface:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.prompt = PromptManager()
        try:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception:
            self.client = None

    def generate_kernels(self, prompt: str, num_candidates: int) -> List[CandidateKernel]:
        """Generate CUDA kernels using structured output parsing"""
        results: List[CandidateKernel] = []

        for i in range(num_candidates):
            kernel_name = f"kernel_{i}"
            try:
                if self.client is None:
                    raise RuntimeError("OpenAI client not initialized")

                system_prompt = """You are an expert CUDA kernel engineer with deep knowledge of GPU architecture and optimization techniques.

Generate high-performance CUDA kernel code that:
1. Uses optimal memory access patterns (coalesced reads/writes)
2. Leverages shared memory to reduce global memory traffic
3. Employs warp-level primitives for efficient thread communication
4. Maximizes occupancy while minimizing register pressure
5. Handles edge cases and boundary conditions properly
6. Is tailored to the specific GPU hardware characteristics

Return a JSON object with the following structure:
{
  "code": "complete .cu file content with kernel implementation",
  "kernel_name": "name of the kernel function",
  "optimizations_used": ["list", "of", "optimization", "techniques", "applied"]
}

Include all necessary headers and ensure the kernel signature matches the requirements."""

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    response_format={
                        "type": "json_object"
                    }
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI API")
                
                import json
                try:
                    data = json.loads(content)
                    kernel_data = GeneratedCUDAKernel(
                        code=data.get("code", ""),
                        kernel_name=data.get("kernel_name", kernel_name),
                        optimizations_used=data.get("optimizations_used", [])
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(f"Failed to parse structured output: {e}")

                # Ensure kernel has proper name and structure
                code = kernel_data.code
                if not code.strip():
                    raise ValueError("Generated code is empty")

                # Validate that code contains essential CUDA elements
                if not any(keyword in code for keyword in ['__global__', '__device__', 'extern "C"']):
                    raise ValueError("Generated code lacks CUDA kernel markers")

                results.append(CandidateKernel(
                    code=code,
                    name=kernel_data.kernel_name or kernel_name,
                    metadata={
                        "source": "openai-structured",
                        "optimizations": getattr(kernel_data, "optimizations_used", []),
                        "generation_attempt": i,
                    },
                ))

            except Exception as e:
                # Fallback to synthetic kernel for this candidate
                results.append(CandidateKernel(
                    code=self._create_fallback_kernel(kernel_name),
                    name=kernel_name,
                    metadata={
                        "source": "synthetic-fallback",
                        "error": str(e),
                        "generation_attempt": i
                    }
                ))

        return results

    def _create_fallback_kernel(self, name: str) -> str:
        """Create a basic but functional CUDA kernel as fallback"""
        return f"""
            #include <cuda_runtime.h>
            #include <device_launch_parameters.h>

            extern "C" __global__ void {name}(const float* __restrict__ input, float* __restrict__ output, int n) {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < n) {{
                    output[idx] = input[idx];
                }}
            }}
            """
