from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, PatternKnowledge
from ..vector_store import search_stage
import os


class PatternRetriever:
    def __init__(self):
        self.api_key = os.environ.get("CHROMA_API_KEY")
        self.tenant = os.environ.get("CHROMA_TENANT")
        self.database = os.environ.get("CHROMA_DATABASE")
        # stage handled via search_stage

    def retrieve(self, request: StructuredUserIntent, context: Dict) -> PatternKnowledge:
        q = f"CUDA optimization patterns for {request.task}. Block size, shared memory tiling, loop unrolling."
        snippets = search_stage(stage="patterns", query=q, k=3)
        return PatternKnowledge(snippets=snippets)
