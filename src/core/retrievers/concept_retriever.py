from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, ConceptKnowledge
from ..vector_store import search_stage
import os


class ConceptRetriever:
    def __init__(self):
        self.api_key = os.environ.get("CHROMA_API_KEY")
        self.tenant = os.environ.get("CHROMA_TENANT")
        self.database = os.environ.get("CHROMA_DATABASE")
        # stage handled via search_stage

    def retrieve(self, request: StructuredUserIntent, context: Dict) -> ConceptKnowledge:
        q = f"CUDA concepts for task: {request.task}. Threading model, memory hierarchy, synchronization, coalescing."
        snippets = search_stage(stage="concepts", query=q, k=3)
        return ConceptKnowledge(snippets=snippets)
