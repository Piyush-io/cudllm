from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, APIKnowledge
from ..vector_store import search_stage
import os


class APIRetriever:
    def __init__(self):
        self.api_key = os.environ.get("CHROMA_API_KEY")
        self.tenant = os.environ.get("CHROMA_TENANT")
        self.database = os.environ.get("CHROMA_DATABASE")
        # stage handled via search_stage

    def retrieve(self, request: StructuredUserIntent, context: Dict) -> APIKnowledge:
        q = "nvcc flags for target arch, cudaEvent timing best practices, cudaMemcpy, streams."
        snippets = search_stage(stage="api", query=q, k=3)
        return APIKnowledge(snippets=snippets)
