from typing import Dict, List
from ..schemas.knowledge import (
    StructuredUserIntent,
    KnowledgeBase,
    ConceptKnowledge,
    PatternKnowledge,
    HardwareKnowledge,
    APIKnowledge,
    ExampleKnowledge,
)
from .retrievers.concept_retriever import ConceptRetriever
from .retrievers.pattern_retriever import PatternRetriever
from .retrievers.hardware_retriever import HardwareRetriever
from .retrievers.api_retriever import APIRetriever
from .retrievers.example_retriever import ExampleRetriever


class HierarchicalRAG:
    def __init__(self) -> None:
        self.concept_retriever = ConceptRetriever()
        self.pattern_retriever = PatternRetriever()
        self.hardware_retriever = HardwareRetriever()
        self.api_retriever = APIRetriever()
        self.example_retriever = ExampleRetriever()

    def retrieve_knowledge(self, request: StructuredUserIntent) -> KnowledgeBase:
        context: Dict = {}
        trace: List[str] = []

        concepts = self.concept_retriever.retrieve(request, context)
        context["concepts"] = concepts
        trace.append("concepts")

        patterns = self.pattern_retriever.retrieve(request, context)
        context["patterns"] = patterns
        trace.append("patterns")

        hardware = self.hardware_retriever.retrieve(request, context)
        context["hardware"] = hardware
        trace.append("hardware")

        api = self.api_retriever.retrieve(request, context)
        context["api"] = api
        trace.append("api")

        examples = self.example_retriever.retrieve(request, context)
        context["examples"] = examples
        trace.append("examples")

        return KnowledgeBase(
            concepts=concepts,
            patterns=patterns,
            hardware=hardware,
            api=api,
            examples=examples,
            retrieval_trace=trace,
        )

    def get_retrieval_summary(self, knowledge_base: KnowledgeBase) -> str:
        parts: List[str] = []
        if knowledge_base.concepts:
            parts.append(f"CUDA Concepts: {len(knowledge_base.concepts.snippets)} snippets")
        if knowledge_base.patterns:
            parts.append(f"Algorithm Patterns: {len(knowledge_base.patterns.snippets)} snippets")
        if knowledge_base.hardware:
            parts.append(f"Hardware Constraints: {len(knowledge_base.hardware.snippets)} snippets")
        if knowledge_base.api:
            parts.append(f"API Knowledge: {len(knowledge_base.api.snippets)} snippets")
        if knowledge_base.examples:
            parts.append(f"Reference Examples: {len(knowledge_base.examples.snippets)} snippets")
        return " | ".join(parts)
