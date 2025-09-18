from typing import List, Optional
from pydantic import BaseModel


class KernelsResponse(BaseModel):
    kernels: List[str]


# Backwards-compat alias if referenced elsewhere
kernels_response = KernelsResponse

__all__ = ["KernelsResponse"]


class CompileResult(BaseModel):
    ok: bool
    binary_path: str | None = None
    stderr: Optional[str] = None


class ValidationResult(BaseModel):
    ok: bool
    output: str


class ProfileResult(BaseModel):
    ok: bool
    time_ms: float
    output: str


class CandidateResult(BaseModel):
    kernel_src: str
    compile: CompileResult
    validatation: Optional[ValidationResult] = None
    profile: Optional[ProfileResult] = None


class FSRSearchResult(BaseModel):
    best_kernel: Optional[str]
    best_time_ms: float
    iterations: int
    candidates: List[CandidateResult]


__all__ += [
    "CompileResult",
    "ValidationResult",
    "ProfileResult",
    "CandidateResult",
    "FSRSearchResult",
]
