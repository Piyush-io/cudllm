"""Example knowledge retriever."""

from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, ExampleKnowledge
from ..vector_store import search_stage


class ExampleRetriever:
    """Retrieves CUDA example code knowledge from vector store."""

    def retrieve(
        self, request: StructuredUserIntent, context: Dict
    ) -> ExampleKnowledge:
        """
        Retrieve example knowledge relevant to the user's request.

        Args:
            request: Structured user intent with task and constraints.
            context: Shared context from previous retrieval stages.

        Returns:
            ExampleKnowledge with relevant snippets.
        """
        q = (
            f"Reference CUDA implementations for {request.task} "
            "with correctness checks and event timing."
        )
        snippets = search_stage(stage="examples", query=q, k=3)
        return ExampleKnowledge(snippets=snippets)
