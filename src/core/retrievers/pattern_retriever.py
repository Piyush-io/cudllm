"""Pattern knowledge retriever."""

from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, PatternKnowledge
from ..vector_store import search_stage


class PatternRetriever:
    """Retrieves CUDA optimization pattern knowledge from vector store."""

    def retrieve(
        self, request: StructuredUserIntent, context: Dict
    ) -> PatternKnowledge:
        """
        Retrieve pattern knowledge relevant to the user's request.

        Args:
            request: Structured user intent with task and constraints.
            context: Shared context from previous retrieval stages.

        Returns:
            PatternKnowledge with relevant snippets.
        """
        q = (
            f"CUDA optimization patterns for {request.task}. "
            "Block size, shared memory tiling, loop unrolling."
        )
        snippets = search_stage(stage="patterns", query=q, k=3)
        return PatternKnowledge(snippets=snippets)
