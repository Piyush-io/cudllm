from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkTask:
    name: str
    description: str
    host_code: str = ""
    test_sizes: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def vector_add_task() -> BenchmarkTask:
    desc = (
        "Write a CUDA C++ program that performs elementwise vector addition for float32 arrays. "
        "Compute C[i] = A[i] + B[i] for N = 1<<20. Use 256 threads per block and a grid sized to cover N. "
        "The program must be self-contained, generate input data, compute a CPU reference, compare results within 1e-5, "
        "print 'OK' if all elements match, otherwise 'FAIL'. Measure GPU kernel time using cudaEvent elapsed time, "
        "after a warmup launch, averaged over multiple runs, and print a single line 'TIME_MS <number>'."
    )
    return BenchmarkTask("vector_add", desc, "", [1 << 20, 1 << 22], ["memory", "elementwise"])


__all__ = ["BenchmarkTask", "vector_add_task"]
