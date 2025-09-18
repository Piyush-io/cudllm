from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, HardwareKnowledge
from ..vector_store import search_stage
import os


class HardwareRetriever:
    def __init__(self):
        self.api_key = os.environ.get("CHROMA_API_KEY")
        self.tenant = os.environ.get("CHROMA_TENANT")
        self.database = os.environ.get("CHROMA_DATABASE")
        # stage handled via search_stage

    def retrieve(self, request: StructuredUserIntent, context: Dict) -> HardwareKnowledge:
        arch = request.hardware_arch or "sm_80"
        q = f"CUDA compute capability {arch} limits and occupancy guidance. Registers/SM, shared memory/SM, warp size."
        snippets = search_stage(stage="hardware", query=q, k=3)
        return HardwareKnowledge(snippets=snippets)
