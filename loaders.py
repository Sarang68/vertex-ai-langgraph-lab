# ingestion/loaders.py
from dataclasses import dataclass, field
from typing import Iterator, Optional
from pathlib import Path
import hashlib, datetime, json
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

@dataclass
class Document:
    """Canonical document with rich metadata for RAG."""
    content: str
    doc_id: str                         # stable hash-based ID
    source: str                         # file path or URL
    source_type: str                    # pdf | web | api
    title: Optional[str] = None
    page_num: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    access_groups: list[str] = field(default_factory=lambda: ["public"])
    metadata: dict = field(default_factory=dict)

    def to_gcs_payload(self) -> dict:
        return {"id": self.doc_id, "content": self.content,
                "metadata": {"source": self.source, "type": self.source_type,
                             "title": self.title, "page": self.page_num,
                             "access_groups": self.access_groups,
                             "created_at": self.created_at, **self.metadata}}

class PDFLoader:
    def __init__(self, access_groups: list[str] = None):
        self.access_groups = access_groups or ["internal"]

    def load(self, path: str) -> Iterator[Document]:
        doc = fitz.open(path)
        file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) < 50:
                continue  # skip near-empty pages
            yield Document(
                content=text,
                doc_id=f"{file_hash}-p{page_num}",
                source=path, source_type="pdf",
                title=doc.metadata.get("title") or Path(path).stem,
                page_num=page_num,
                access_groups=self.access_groups,
                metadata={"total_pages": len(doc), "author": doc.metadata.get("author")}
            )

class WebLoader:
    def load(self, url: str, access_groups=None) -> Document:
        resp = requests.get(url, timeout=15,
            headers={"User-Agent": "RAG-Pipeline/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = " ".join(soup.get_text().split())
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return Document(
            content=text, doc_id=f"web-{url_hash}", source=url,
            source_type="web", title=soup.title.string if soup.title else url,
            access_groups=access_groups or ["public"]
        )
