# retrieval/hybrid_search.py
from dataclasses import dataclass
from typing import Optional
from rank_bm25 import BM25Okapi
import numpy as np

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    metadata: dict
    retrieval_method: str  # dense | sparse | hybrid

class HybridRetriever:
    """
    Dense (Vertex Vector Search) + Sparse (BM25) with
    Reciprocal Rank Fusion. Alpha controls dense/sparse balance.
    """
    def __init__(self, vector_index, embedder, corpus_chunks: list, alpha: float = 0.7):
        self.vector_index = vector_index
        self.embedder = embedder
        self.alpha = alpha  # 1.0 = pure dense, 0.0 = pure sparse
        # Build BM25 index over corpus
        tokenized = [c.text.lower().split() for c in corpus_chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = corpus_chunks

    def _reciprocal_rank_fusion(
        self, dense_hits: list, sparse_hits: list, k: int = 60
    ) -> list[tuple[str, float]]:
        """RRF(d) = Σ 1/(k + rank(d)). k=60 is Cormack et al.'s recommendation."""
        scores: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1 / (k + rank + 1)
        for rank, hit in enumerate(sparse_hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 20,
                  access_groups: list[str] = None) -> list[RetrievedChunk]:
        # Dense retrieval
        q_emb = self.embedder.embed_query(query)
        dense_results = self.vector_index.find_neighbors(
            queries=[q_emb], num_neighbors=top_k,
            restricts=[{"namespace": "access_group",
                        "allow_list": access_groups or ["public"]}]
        )[0]

        # Sparse retrieval (BM25)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_sparse_idx = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_results = [self.chunks[i] for i in top_sparse_idx]

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        chunk_map = {c.chunk_id: c for c in self.chunks}
        return [
            RetrievedChunk(chunk_id=cid, text=chunk_map[cid].text,
                            score=score, doc_id=chunk_map[cid].doc_id,
                            metadata=chunk_map[cid].metadata,
                            retrieval_method="hybrid")
            for cid, score in fused[:top_k] if cid in chunk_map
        ]
