# ingestion/gcs_sink.py
from google.cloud import storage, pubsub_v1
import json, os

BUCKET = os.environ["GCS_BUCKET"]        # e.g. rag-lab-docs
TOPIC  = os.environ["PUBSUB_TOPIC"]       # projects/{id}/topics/doc-ingested

class GCSSink:
    def __init__(self):
        self.gcs     = storage.Client()
        self.bucket  = self.gcs.bucket(BUCKET)
        self.pub     = pubsub_v1.PublisherClient()

    def write(self, doc) -> str:
        blob_path = f"raw/{doc.source_type}/{doc.doc_id}.json"
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(doc.to_gcs_payload(), ensure_ascii=False),
            content_type="application/json"
        )
        # Fire Pub/Sub event so downstream embedding worker can pick up
        self.pub.publish(TOPIC, json.dumps({
            "doc_id": doc.doc_id,
            "blob_path": blob_path,
            "source_type": doc.source_type
        }).encode())
        return blob_path

# ── Usage ────────────────────────────────────────────
from loaders import PDFLoader
sink = GCSSink()
for doc in PDFLoader(access_groups=["finance-team"]).load("report.pdf"):
    path = sink.write(doc)
    print(f"Staged: {path}")
