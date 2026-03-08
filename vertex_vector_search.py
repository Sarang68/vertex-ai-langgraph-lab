# vector_db/vertex_vector_search.py
from google.cloud import aiplatform
import json, os
from pathlib import Path

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
REGION  = "us-central1"
BUCKET  = os.environ["GCS_BUCKET"]

aiplatform.init(project=PROJECT, location=REGION)

class VertexVectorIndex:
    def create_index(self, display_name: str, dimensions: int = 768):
        """Create a streaming-update HNSW index."""
        self.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=display_name,
            dimensions=dimensions,
            approximate_neighbors_count=150,        # balance recall vs speed
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=7,
            distance_measure_type="DOT_PRODUCT_DISTANCE",  # for normalized vecs
            index_update_method="STREAM_UPDATE",    # real-time upserts
        )
        print(f"Index created: {self.index.resource_name}")
        return self.index

    def deploy_endpoint(self, display_name: str):
        """Deploy index to a public or private endpoint."""
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=f"{display_name}-endpoint",
            public_endpoint_enabled=True          # use VPC for production
        )
        self.endpoint.deploy_index(
            index=self.index,
            deployed_index_id=display_name.replace("-", "_"),
            min_replica_count=1,
            max_replica_count=5,          # auto-scale for variable load
        )
        return self.endpoint

    def upsert_batch(self, chunks: list, embeddings: list[list[float]]):
        """Stream upsert chunks with embeddings + metadata."""
        datapoints = []
        for chunk, emb in zip(chunks, embeddings):
            datapoints.append(
                aiplatform.MatchingEngineIndex.Datapoint(
                    datapoint_id=chunk.chunk_id,
                    feature_vector=emb,
                    restricts=[
                        aiplatform.MatchingEngineIndex.Datapoint.Restriction(
                            namespace="access_group",
                            allow_list=chunk.metadata.get("access_groups", ["public"])
                        )
                    ]
                )
            )
        # Upsert in batches of 100 (API limit)
        for i in range(0, len(datapoints), 100):
            self.index.upsert_datapoints(datapoints[i:i+100])
