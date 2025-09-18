from typing import List, Dict, Optional
from pydantic import BaseModel


class StructuredUserIntent(BaseModel):
    task: str
    constraints: Dict[str, str] = {}
    hardware_arch: Optional[str] = None
    perf_goal: Optional[str] = None


class ConceptKnowledge(BaseModel):
    snippets: List[str]


class PatternKnowledge(BaseModel):
    snippets: List[str]


class HardwareKnowledge(BaseModel):
    snippets: List[str]


class APIKnowledge(BaseModel):
    snippets: List[str]


class ExampleKnowledge(BaseModel):
    snippets: List[str]


class KnowledgeBase(BaseModel):
    concepts: Optional[ConceptKnowledge] = None
    patterns: Optional[PatternKnowledge] = None
    hardware: Optional[HardwareKnowledge] = None
    api: Optional[APIKnowledge] = None
    examples: Optional[ExampleKnowledge] = None
    retrieval_trace: List[str] = []
