# api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.gcp.trace import CloudTraceSpanExporter
import structlog, time

# ── OpenTelemetry Setup ────────────────────────────────
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(CloudTraceSpanExporter())
)
tracer = trace.get_tracer("rag-pipeline")

# ── Structured Logging ─────────────────────────────────
log = structlog.get_logger()

app = FastAPI(title="RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    user_id: str
    access_groups: list[str] = ["public"]
    max_tokens: int = 1024

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    latency_ms: float
    tokens_used: int
    trace_id: str

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    with tracer.start_as_current_span("rag-query") as span:
        t0 = time.perf_counter()
        span.set_attribute("query.length", len(req.query))
        span.set_attribute("user.access_groups", str(req.access_groups))

        try:
            # Retrieval span
            with tracer.start_as_current_span("retrieval"):
                chunks = retriever.retrieve(req.query, access_groups=req.access_groups)
                span.set_attribute("retrieval.chunks_count", len(chunks))

            # Generation span
            with tracer.start_as_current_span("generation"):
                result = generator.generate(req.query, chunks)

            latency = (time.perf_counter() - t0) * 1000
            log.info("query_success", latency_ms=round(latency, 2),
                     grounded=result.grounded, user=req.user_id)

            return QueryResponse(
                answer=result.answer, citations=result.citations,
                latency_ms=latency, tokens_used=0,
                trace_id=format(span.get_span_context().trace_id, "032x")
            )
        except Exception as e:
            log.error("query_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
