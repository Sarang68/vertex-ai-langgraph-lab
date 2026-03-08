# evaluation/metrics.py
import numpy as np
from vertexai.generative_models import GenerativeModel

# ── Retrieval Metrics ──────────────────────────────────
def mean_reciprocal_rank(relevant_ids: set, retrieved_ids: list) -> float:
    """MRR: 1/rank of first relevant result. Good for "find one answer" tasks."""
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(relevant_ids: set, retrieved_ids: list, k: int = 10) -> float:
    """NDCG@k: ranking-aware metric. Better than MRR when multiple relevant docs exist."""
    def dcg(ids):
        return sum(1 / np.log2(i + 2) for i, rid in enumerate(ids[:k]) if rid in relevant_ids)
    ideal = dcg(list(relevant_ids)[:k])
    return dcg(retrieved_ids) / ideal if ideal > 0 else 0.0

def context_recall_manual(ground_truth: str, retrieved_texts: list[str]) -> float:
    """Fraction of ground truth statements supported by retrieved context."""
    return sum(1 for t in retrieved_texts
                if any(kw in t for kw in ground_truth.split()[:10])) / len(retrieved_texts) if retrieved_texts else 0.0

# ── LLM-as-Judge ──────────────────────────────────────
class LLMJudge:
    RUBRIC = """Score this RAG answer on a 1-5 scale for each dimension:
    1. Faithfulness: Is the answer fully supported by the context (no hallucination)?
    2. Completeness: Does it address all aspects of the question?
    3. Conciseness: Is it appropriately concise without omitting key info?

    Return JSON: {{"faithfulness": 1-5, "completeness": 1-5, "conciseness": 1-5, "reasoning": "..."}}

    QUESTION: {question}
    CONTEXT: {context}
    ANSWER: {answer}"""

    def __init__(self):
        self.model = GenerativeModel("gemini-1.5-flash")  # cheaper for eval

    def score(self, question: str, context: str, answer: str) -> dict:
        import json
        resp = self.model.generate_content(
            self.RUBRIC.format(question=question, context=context[:3000], answer=answer),
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(resp.text)
