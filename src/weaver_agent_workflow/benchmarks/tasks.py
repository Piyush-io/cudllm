from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class BenchmarkTask:
    name: str
    description: str
    host_code: str
    test_sizes: List[int]
    reference_implementation: str = "identity"

    def generate_test_cases(self) -> Dict[str, Any]:
        sizes = self.test_sizes
        inputs = [np.random.rand(s).astype(np.float32) for s in sizes]
        return {"inputs": inputs}

    def get_reference_output(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        if self.reference_implementation == "identity":
            return [x.copy() for x in inputs]
        elif self.reference_implementation == "vector_add":
            return [x + x for x in inputs]
        elif self.reference_implementation == "scale":
            return [x * 2.0 for x in inputs]
        elif self.reference_implementation == "square":
            return [x * x for x in inputs]
        else:
            return [x.copy() for x in inputs]


class BenchmarkSuite:
    @staticmethod
    def get_identity_task() -> BenchmarkTask:
        return BenchmarkTask(
            name="identity",
            description="Copy input array to output array (identity operation)",
            host_code="""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __global__ void kernel(const float* input, float* output, int n);

// Test harness code - do not modify
extern "C" void launch_kernel(float* input, float* output, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    kernel<<<grid, block>>>(input, output, n);
}
            """,
            test_sizes=[1024, 4096, 16384],
            reference_implementation="identity"
        )

    @staticmethod
    def get_vector_add_task() -> BenchmarkTask:
        return BenchmarkTask(
            name="vector_add",
            description="Add each element of input array to itself (output[i] = input[i] + input[i])",
            host_code="""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __global__ void kernel(const float* input, float* output, int n);

// Test harness code - do not modify
extern "C" void launch_kernel(float* input, float* output, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    kernel<<<grid, block>>>(input, output, n);
}
            """,
            test_sizes=[1024, 4096, 16384],
            reference_implementation="vector_add"
        )

    @staticmethod
    def get_scale_task() -> BenchmarkTask:
        return BenchmarkTask(
            name="scale",
            description="Scale input array by factor of 2.0 (output[i] = input[i] * 2.0)",
            host_code="""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __global__ void kernel(const float* input, float* output, int n);

// Test harness code - do not modify  
extern "C" void launch_kernel(float* input, float* output, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    kernel<<<grid, block>>>(input, output, n);
}
            """,
            test_sizes=[1024, 4096, 16384],
            reference_implementation="scale"
        )

    @staticmethod
    def get_square_task() -> BenchmarkTask:
        return BenchmarkTask(
            name="square",
            description="Square each element of input array (output[i] = input[i] * input[i])",
            host_code="""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" __global__ void kernel(const float* input, float* output, int n);

// Test harness code - do not modify
extern "C" void launch_kernel(float* input, float* output, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    kernel<<<grid, block>>>(input, output, n);
}
            """,
            test_sizes=[1024, 4096, 16384],
            reference_implementation="square"
        )

    @staticmethod
    def get_all_tasks() -> List[BenchmarkTask]:
        return [
            BenchmarkSuite.get_identity_task(),
            BenchmarkSuite.get_vector_add_task(),
            BenchmarkSuite.get_scale_task(),
            BenchmarkSuite.get_square_task()
        ]
