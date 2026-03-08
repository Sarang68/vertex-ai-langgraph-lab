# embeddings/vertex_embedder.py
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import asyncio
from typing import Literal

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY",
                   "SEMANTIC_SIMILARITY", "CLASSIFICATION"]

class VertexEmbedder:
    """
    Wraps text-embedding-004 with batching, task-type awareness,
    and dimensionality control (256 / 768 / 1024).
    """
    MODEL   = "text-embedding-004"
    BATCH   = 250   # Vertex AI hard limit per request
    MAX_TOK = 2048  # tokens per input

    def __init__(self, output_dimensionality: int = 768):
        self.model = TextEmbeddingModel.from_pretrained(self.MODEL)
        self.dim = output_dimensionality

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Use RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY at search time
        inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in texts]
        all_embeddings = []
        for i in range(0, len(inputs), self.BATCH):
            batch = inputs[i:i + self.BATCH]
            results = self.model.get_embeddings(
                batch,
                output_dimensionality=self.dim,
                auto_truncate=True          # silently truncate > MAX_TOK
            )
            all_embeddings.extend([r.values for r in results])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.model.get_embeddings(
            [TextEmbeddingInput(text, "RETRIEVAL_QUERY")],
            output_dimensionality=self.dim
        )
        return result[0].values

# ── Interview insight: task types matter! ─────────────────────
# RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY produces asymmetric embeddings
# optimized for the bi-encoder retrieval pattern. Using RETRIEVAL_QUERY
# at index time is a common mistake that degrades retrieval quality ~15%.
