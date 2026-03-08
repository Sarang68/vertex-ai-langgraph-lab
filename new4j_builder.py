# graph/neo4j_builder.py
from neo4j import GraphDatabase
from vertexai.generative_models import GenerativeModel
import json, os

class GraphRAGBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASS"])
        )
        self.model = GenerativeModel("gemini-1.5-flash")

    def extract_entities(self, text: str) -> dict:
        """Use Gemini to extract entities + relationships from text."""
        prompt = f"""Extract entities and relationships from this text.
Return JSON: {{"entities": [{{"id": "...", "type": "Person|Org|Product|Policy", "name": "..."}}],
               "relationships": [{{"from": "id", "to": "id", "type": "WORKS_AT|GOVERNS|APPLIES_TO"}}]}}
TEXT: {text[:2000]}"""
        resp = self.model.generate_content(prompt,
            generation_config={"response_mime_type": "application/json"})
        return json.loads(resp.text)

    def ingest_to_graph(self, chunk_id: str, entities: dict):
        with self.driver.session() as s:
            # Merge entities (upsert pattern)
            for e in entities["entities"]:
                s.run("""MERGE (n {id: $id})
                    SET n.name = $name, n.type = $type
                    MERGE (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(n)""",
                    id=e["id"], name=e["name"], type=e["type"], chunk_id=chunk_id)
            # Merge relationships
            for r in entities["relationships"]:
                s.run(f"""MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                    MERGE (a)-[:{r['type']}]->(b)""",
                    from_id=r["from"], to_id=r["to"])

    def graph_retrieve(self, entity_name: str, hops: int = 2) -> list[str]:
        """Multi-hop graph traversal — returns related chunk IDs."""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (start {name: $name})
                MATCH path = (start)-[*1..$hops]-(related)
                MATCH (c:Chunk)-[:MENTIONS]->(related)
                RETURN DISTINCT c.id AS chunk_id, length(path) AS hops
                ORDER BY hops LIMIT 20
            """, name=entity_name, hops=hops)
            return [r["chunk_id"] for r in result]

# ── Graph RAG Fusion ──────────────────────────────────
def graph_rag_retrieve(query: str, hybrid_retriever, graph_builder,
                        alpha: float = 0.6) -> list:
    """Fuse vector retrieval + graph traversal."""
    # Vector retrieval
    vector_chunks = hybrid_retriever.retrieve(query, top_k=15)

    # Extract query entities and graph-traverse
    query_entities = graph_builder.extract_entities(query)
    graph_chunk_ids = []
    for e in query_entities.get("entities", []):
        graph_chunk_ids.extend(graph_builder.graph_retrieve(e["name"]))

    # Boost vector chunks that also appear in graph results
    boosted = []
    graph_set = set(graph_chunk_ids)
    for chunk in vector_chunks:
        boost = 1.3 if chunk.chunk_id in graph_set else 1.0
        chunk.score *= boost
        boosted.append(chunk)
    return sorted(boosted, key=lambda x: x.score, reverse=True)
