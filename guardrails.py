# cost/guardrails.py
from functools import lru_cache
import hashlib, time

class CostGuardrails:
    # Gemini 1.5 Pro pricing: $3.50/1M input tokens (as of 2024)
    TOKEN_COST_PER_M = 3.50
    DAILY_BUDGET_USD = 50.0

    def __init__(self, cache_ttl: int = 3600):
        self._cache: dict = {}
        self._daily_tokens: int = 0
        self.cache_ttl = cache_ttl

    def cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def check_cache(self, query: str) -> dict | None:
        key = self.cache_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < self.cache_ttl:
                return entry["result"]  # Cache hit → $0 cost
        return None

    def enforce_budget(self, estimated_tokens: int):
        cost = (self._daily_tokens + estimated_tokens) / 1_000_000 * self.TOKEN_COST_PER_M
        if cost > self.DAILY_BUDGET_USD:
            raise ValueError(f"Daily budget ${self.DAILY_BUDGET_USD} would be exceeded")
        self._daily_tokens += estimated_tokens
