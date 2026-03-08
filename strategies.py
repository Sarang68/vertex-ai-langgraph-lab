# chunking/strategies.py
from typing import Protocol, Iterator
from dataclasses import dataclass
import tiktoken, re
import numpy as np

@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    token_count: int
    metadata: dict

# ── Strategy 1: Fixed-Size Token Chunking ─────────────
class FixedTokenChunker:
    def __init__(self, max_tokens=512, overlap=128):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, doc) -> Iterator[Chunk]:
        tokens = self.enc.encode(doc.content)
        step = self.max_tokens - self.overlap
        for i, start in enumerate(range(0, len(tokens), step)):
            chunk_tokens = tokens[start:start + self.max_tokens]
            yield Chunk(
                text=self.enc.decode(chunk_tokens),
                chunk_id=f"{doc.doc_id}-c{i}",
                doc_id=doc.doc_id, chunk_index=i,
                token_count=len(chunk_tokens),
                metadata={**doc.metadata, "strategy": "fixed"}
            )

# ── Strategy 2: Recursive Semantic Chunking ───────────
class RecursiveSemanticChunker:
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, max_tokens=400, overlap_tokens=80):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        splits = text.split(sep) if sep else list(text)
        chunks, current = [], ""
        for s in splits:
            candidate = (current + sep + s).strip()
            if len(self.enc.encode(candidate)) <= self.max_tokens:
                current = candidate
            else:
                if current: chunks.append(current)
                if len(self.enc.encode(s)) > self.max_tokens:
                    chunks.extend(self._split_text(s, separators[1:]))
                    current = ""
                else: current = s
        if current: chunks.append(current)
        return chunks

    def chunk(self, doc) -> list[Chunk]:
        splits = self._split_text(doc.content, self.SEPARATORS)
        return [Chunk(text=t, chunk_id=f"{doc.doc_id}-c{i}",
                       doc_id=doc.doc_id, chunk_index=i,
                       token_count=len(self.enc.encode(t)),
                       metadata={**doc.metadata, "strategy": "recursive"})
                for i, t in enumerate(splits)]
