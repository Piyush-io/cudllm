"""Hardware knowledge retriever."""

from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, HardwareKnowledge
from ..vector_store import search_stage


class HardwareRetriever:
    """Retrieves hardware-specific CUDA knowledge from vector store."""

    def retrieve(
        self, request: StructuredUserIntent, context: Dict
    ) -> HardwareKnowledge:
        """
        Retrieve hardware knowledge relevant to the user's request.

        Args:
            request: Structured user intent with task and constraints.
            context: Shared context from previous retrieval stages.

        Returns:
            HardwareKnowledge with relevant snippets.
        """
        arch = request.hardware_arch or "sm_80"
        q = (
            f"CUDA compute capability {arch} limits and occupancy guidance. "
            "Registers/SM, shared memory/SM, warp size."
        )
        snippets = search_stage(stage="hardware", query=q, k=3)
        return HardwareKnowledge(snippets=snippets)
