# retrieval/generator.py
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from pydantic import BaseModel

class CitedAnswer(BaseModel):
    answer: str
    citations: list[dict]   # [{chunk_id, doc_id, excerpt}]
    confidence: str         # high | medium | low
    grounded: bool          # could answer be verified from context?

class GroundedGenerator:
    def __init__(self):
        self.model = GenerativeModel("gemini-1.5-pro")

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> CitedAnswer:
        context = "\n\n".join([
            f"[SOURCE {i+1} | id:{c.chunk_id}]\n{c.text}"
            for i, c in enumerate(chunks[:8])  # top-8 for context window
        ])

        prompt = f"""You are a precise RAG assistant. Answer ONLY from the sources.
If the answer cannot be found in sources, say so explicitly.

SOURCES:
{context}

QUESTION: {query}

Respond in JSON:
{{
  "answer": "...",
  "citations": [{{"source_num": 1, "chunk_id": "...", "excerpt": "..."}}],
  "confidence": "high|medium|low",
  "grounded": true|false
}}"""

        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",  # structured output
                temperature=0.1,  max_output_tokens=1024
            )
        )
        import json
        data = json.loads(response.text)
        return CitedAnswer(**data)
