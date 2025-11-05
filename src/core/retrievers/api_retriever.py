"""API knowledge retriever."""

from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, APIKnowledge
from ..vector_store import search_stage


class APIRetriever:
    """Retrieves CUDA API knowledge from vector store."""

    def retrieve(self, request: StructuredUserIntent, context: Dict) -> APIKnowledge:
        """
        Retrieve API knowledge relevant to the user's request.

        Args:
            request: Structured user intent with task and constraints.
            context: Shared context from previous retrieval stages.

        Returns:
            APIKnowledge with relevant snippets.
        """
        q = (
            "nvcc flags for target arch, cudaEvent timing best practices, "
            "cudaMemcpy, streams."
        )
        snippets = search_stage(stage="api", query=q, k=3)
        return APIKnowledge(snippets=snippets)
