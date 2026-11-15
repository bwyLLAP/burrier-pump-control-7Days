from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict


SUPPORTED_SUFFIXES = {".py", ".md", ".json", ".txt"}
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".vendor",
    ".runtime",
    "audit",
    "cache",
    "outputs",
    "pump-full-test-tmp",
    "final-test-tmp",
    "cbc-final-verification-tmp",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: Path
    category: str
    authority_rank: int


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: str
    source_category: str
    authority_rank: int
    chunk_id: str
    content_hash: str
    text: str


class EvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: str
    source_category: str
    chunk_id: str
    excerpt: str


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def _iter_chunks(text: str, max_chars: int = 1400) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    pending = ""
    for block in blocks:
        if len(block) > max_chars:
            if pending:
                chunks.append(pending)
                pending = ""
            lines = block.splitlines()
            segment = ""
            for line in lines:
                candidate = f"{segment}\n{line}".strip()
                if segment and len(candidate) > max_chars:
                    chunks.append(segment)
                    segment = line
                else:
                    segment = candidate
            if segment:
                chunks.append(segment)
            continue
        candidate = f"{pending}\n\n{block}".strip()
        if pending and len(candidate) > max_chars:
            chunks.append(pending)
            pending = block
        else:
            pending = candidate
    if pending:
        chunks.append(pending)
    return chunks


class LocalKnowledgeIndex:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = list(chunks)
        self._token_sets = [_tokens(chunk.text) for chunk in self.chunks]

    @classmethod
    def build(cls, sources: list[KnowledgeSource]) -> "LocalKnowledgeIndex":
        chunks: list[KnowledgeChunk] = []
        for source in sources:
            root = Path(source.root)
            files = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in files:
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if any(part.lower() in EXCLUDED_PARTS for part in path.parts):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for index, block in enumerate(_iter_chunks(text)):
                    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
                    chunks.append(
                        KnowledgeChunk(
                            source_path=str(path.resolve()),
                            source_category=source.category,
                            authority_rank=source.authority_rank,
                            chunk_id=f"{digest[:12]}-{index}",
                            content_hash=digest,
                            text=block,
                        )
                    )
        return cls(chunks)

    def search(self, query: str, limit: int = 5) -> list[KnowledgeChunk]:
        if limit <= 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        ranked: list[tuple[int, int, str, str, KnowledgeChunk]] = []
        for chunk, chunk_tokens in zip(self.chunks, self._token_sets):
            overlap = len(query_tokens & chunk_tokens)
            if overlap == 0:
                continue
            ranked.append(
                (
                    -overlap,
                    chunk.authority_rank,
                    chunk.source_path.lower(),
                    chunk.chunk_id,
                    chunk,
                )
            )
        ranked.sort(key=lambda item: item[:4])
        return [item[4] for item in ranked[:limit]]

    @staticmethod
    def evidence(chunk: KnowledgeChunk, max_chars: int = 240) -> EvidenceReference:
        excerpt = " ".join(chunk.text.split())
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 1].rstrip() + "…"
        return EvidenceReference(
            source_path=chunk.source_path,
            source_category=chunk.source_category,
            chunk_id=chunk.chunk_id,
            excerpt=excerpt,
        )
