"""Concept knowledge retriever."""

from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, ConceptKnowledge
from ..vector_store import search_stage


class ConceptRetriever:
    """Retrieves CUDA concept knowledge from vector store."""

    def retrieve(
        self, request: StructuredUserIntent, context: Dict
    ) -> ConceptKnowledge:
        """
        Retrieve concept knowledge relevant to the user's request.

        Args:
            request: Structured user intent with task and constraints.
            context: Shared context from previous retrieval stages.

        Returns:
            ConceptKnowledge with relevant snippets.
        """
        q = (
            f"CUDA concepts for task: {request.task}. "
            "Threading model, memory hierarchy, synchronization, coalescing."
        )
        snippets = search_stage(stage="concepts", query=q, k=3)
        return ConceptKnowledge(snippets=snippets)
