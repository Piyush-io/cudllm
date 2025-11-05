"""Knowledge schemas with proper Pydantic defaults."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class StructuredUserIntent(BaseModel):
    """User's intent with task description and constraints."""

    task: str
    constraints: Dict[str, str] = Field(default_factory=dict)
    hardware_arch: Optional[str] = None
    perf_goal: Optional[str] = None


class ConceptKnowledge(BaseModel):
    """CUDA concept knowledge snippets."""

    snippets: List[str] = Field(default_factory=list)


class PatternKnowledge(BaseModel):
    """CUDA optimization pattern snippets."""

    snippets: List[str] = Field(default_factory=list)


class HardwareKnowledge(BaseModel):
    """Hardware-specific CUDA knowledge snippets."""

    snippets: List[str] = Field(default_factory=list)


class APIKnowledge(BaseModel):
    """CUDA API knowledge snippets."""

    snippets: List[str] = Field(default_factory=list)


class ExampleKnowledge(BaseModel):
    """CUDA example code snippets."""

    snippets: List[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    """Complete knowledge base from hierarchical retrieval."""

    concepts: Optional[ConceptKnowledge] = None
    patterns: Optional[PatternKnowledge] = None
    hardware: Optional[HardwareKnowledge] = None
    api: Optional[APIKnowledge] = None
    examples: Optional[ExampleKnowledge] = None
    retrieval_trace: List[str] = Field(default_factory=list)
