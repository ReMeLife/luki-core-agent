"""
Project Knowledge Base using ChromaDB

- One-time ingestion at deploy/startup from configured directories
- Fast semantic search at request time (independent of user ELR)

Env Vars:
- LUKI_PROJECT_KB_PATHS: semi-colon or comma separated absolute/relative dirs
- LUKI_PROJECT_KB_CHROMA_PATH: persistence directory for Chroma (default: ./chroma_project_kb)
- LUKI_PROJECT_KB_EMBED_MODEL: sentence-transformers model (default: all-MiniLM-L6-v2)
- LUKI_PROJECT_KB_TOPK: default top-k results (default: 5)
- LUKI_PROJECT_KB_REBUILD: 'true' to force re-indexing on startup (default: false)
"""
from __future__ import annotations

import os
import hashlib
import logging
from typing import List, Dict, Optional, Mapping, Sequence, Union, Any, cast

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".md", ".txt", ".rst", ".json", ".yaml", ".yml"}


def _iter_files(dirs: List[str]) -> List[str]:
    paths: List[str] = []
    for d in dirs:
        if not d:
            continue
        if not os.path.isdir(d):
            logger.warning(f"ProjectKB: directory not found: {d}")
            continue
        for root, _, files in os.walk(d):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTS:
                    paths.append(os.path.join(root, f))
    return paths


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception as e:
        logger.warning(f"ProjectKB: failed to read {path}: {e}")
        return ""


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        # try to end at a sentence boundary
        if end < n:
            last_period = chunk.rfind(".")
            if last_period > max_chars // 2:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1
        chunks.append(chunk.strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    # filter empty
    return [c for c in chunks if len(c) > 0]


class ProjectKB:
    def __init__(
        self,
        persist_path: Optional[str] = None,
        source_dirs: Optional[List[str]] = None,
        embed_model: str = "all-MiniLM-L6-v2",
        rebuild: bool = False,
        collection_name: str = "project_kb",
    ) -> None:
        self.persist_path = persist_path or os.getenv("LUKI_PROJECT_KB_CHROMA_PATH", "./chroma_project_kb")
        self.source_dirs = source_dirs or []
        self.embed_model = os.getenv("LUKI_PROJECT_KB_EMBED_MODEL", embed_model)
        self.top_k_default = int(os.getenv("LUKI_PROJECT_KB_TOPK", "5"))
        self.collection_name = collection_name
        self.rebuild = os.getenv("LUKI_PROJECT_KB_REBUILD", "false").lower() == "true" or rebuild

        os.makedirs(self.persist_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model
        )
        # get or create
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=cast(Any, self.embed_fn),  # chromadb typing mismatch
            metadata={"hnsw:space": "cosine"},
        )

    def _doc_id(self, path: str, idx: int) -> str:
        h = hashlib.sha256(f"{path}::{idx}".encode("utf-8")).hexdigest()[:32]
        return f"kb_{h}"

    def _already_indexed(self) -> bool:
        try:
            # quick check by counting nr. of items
            count = self.collection.count() or 0
            return count > 0
        except Exception:
            return False

    def ingest(self) -> None:
        if not self.source_dirs:
            logger.info("ProjectKB: no source dirs configured; skipping ingest")
            return
        if self._already_indexed() and not self.rebuild:
            logger.info("ProjectKB: existing index found; skipping ingest")
            return
        if self.rebuild and self._already_indexed():
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=cast(Any, self.embed_fn),  # chromadb typing mismatch
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning(f"ProjectKB: failed to reset collection: {e}")

        files = _iter_files(self.source_dirs)
        logger.info(f"ProjectKB: ingesting {len(files)} files from {self.source_dirs}")
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Union[str, int, float, bool, None]]] = []
        for path in files:
            text = _read_text(path)
            for idx, chunk in enumerate(_chunk_text(text)):
                ids.append(self._doc_id(path, idx))
                docs.append(chunk)
                metas.append({"path": path, "chunk": int(idx)})
            # flush in batches to reduce memory
            if len(ids) >= 256:
                try:
                    self.collection.add(ids=ids, documents=docs, metadatas=cast(Any, metas))
                except Exception as e:
                    logger.warning(f"ProjectKB: add batch failed: {e}")
                ids, docs, metas = [], [], []
        if ids:
            try:
                self.collection.add(ids=ids, documents=docs, metadatas=cast(Any, metas))
            except Exception as e:
                logger.warning(f"ProjectKB: final add batch failed: {e}")
        logger.info("ProjectKB: ingest complete")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        if not query or not query.strip():
            return []
        k = top_k or self.top_k_default
        try:
            res = self.collection.query(query_texts=[query], n_results=k)
            # Chroma may return {"documents": None} or [] when no results
            docs = res.get("documents") or [[]]
            if not isinstance(docs, list):
                docs = [[]]
            first = docs[0] if (len(docs) > 0 and isinstance(docs[0], list)) else []
            out: List[Dict[str, str]] = []
            for d in first:
                if d:
                    out.append({"content": d})
            return out
        except Exception as e:
            logger.warning(f"ProjectKB: search failed: {e}")
            return []
