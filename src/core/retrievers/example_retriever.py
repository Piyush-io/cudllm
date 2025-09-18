from typing import Dict
from ...schemas.knowledge import StructuredUserIntent, ExampleKnowledge
from ..vector_store import search_stage
import os


class ExampleRetriever:
    def __init__(self):
        self.api_key = os.environ.get("CHROMA_API_KEY")
        self.tenant = os.environ.get("CHROMA_TENANT")
        self.database = os.environ.get("CHROMA_DATABASE")
        # stage handled via search_stage

    def retrieve(self, request: StructuredUserIntent, context: Dict) -> ExampleKnowledge:
        q = f"Reference CUDA implementations for {request.task} with correctness checks and event timing."
        snippets = search_stage(stage="examples", query=q, k=3)
        return ExampleKnowledge(snippets=snippets)
