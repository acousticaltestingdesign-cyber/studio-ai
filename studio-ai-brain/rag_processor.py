"""Local gear-manual index for studio-ai-brain.

Scans manuals/ for PDF files, extracts plain text, and writes a JSON vector
index to data/rag_index.json. Vectors are signed feature hashes, so the
index stays in one file and does not need a database server or a downloaded
embedding model.

Run a scan from any working directory:

    python rag_processor.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pypdf import PdfReader

logger = logging.getLogger("studio_ai_brain.rag")

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_DIMENSIONS = 256
MAX_PDF_BYTES = 40 * 1024 * 1024
MAX_PAGES = 400
MAX_CHUNKS = 1200
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
_LOCK = threading.Lock()
_PDF_EXT = r"\.[Pp][Dd][Ff]"
_MANUAL_NAME = re.compile(rf"^[A-Za-z0-9][A-Za-z0-9._ -]{{0,120}}{_PDF_EXT}$")
_SOURCE_NAME = re.compile(rf"^manuals/[A-Za-z0-9][A-Za-z0-9._ -]{{0,120}}{_PDF_EXT}$")
_TOKEN = re.compile(r"[a-z0-9]+")
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


class RagIndexError(Exception):
    def __init__(self) -> None:
        super().__init__("The local manual index could not be read.")


class ManualNotFoundError(Exception):
    def __init__(self, filename: str) -> None:
        super().__init__(f"Manual was not found in manuals/: {filename}")
        self.filename = filename


class ManualParseError(Exception):
    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(reason)
        self.filename = filename
        self.reason = reason


class IndexedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(pattern=_SOURCE_NAME.pattern)
    byte_size: int = Field(ge=0)
    modified_ns: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(ge=0)


class RagChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source: str = Field(pattern=_SOURCE_NAME.pattern)
    page: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=4000)
    vector: list[float] = Field(min_length=VECTOR_DIMENSIONS, max_length=VECTOR_DIMENSIONS)

    @field_validator("vector")
    @classmethod
    def _finite(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(item) for item in value):
            raise ValueError("vector values must be finite")
        return value


class RagIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(pattern=r"^1\.0\.0$")
    embedding: str = Field(pattern=r"^signed_hash_v1$")
    vector_dimensions: int = Field(ge=VECTOR_DIMENSIONS, le=VECTOR_DIMENSIONS)
    sources: list[IndexedSource]
    chunks: list[RagChunk]

    @model_validator(mode="after")
    def _consistent(self) -> RagIndex:
        names = [source.source for source in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("duplicate manual source")
        known = set(names)
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("duplicate chunk id")
        counts = {name: 0 for name in names}
        for chunk in self.chunks:
            if chunk.source not in known:
                raise ValueError("chunk source is not indexed")
            if len(chunk.vector) != self.vector_dimensions:
                raise ValueError("chunk vector has the wrong length")
            counts[chunk.source] += 1
        for source in self.sources:
            if source.chunk_count != counts[source.source]:
                raise ValueError("chunk_count does not match the stored chunks")
        return self


class IngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indexed: list[str]
    unchanged: list[str]
    removed: list[str]
    warnings: list[str]
    errors: list[str]
    chunk_count: int = Field(ge=0)
    index_path: str


@dataclass(frozen=True)
class ManualHit:
    source: str
    page: int
    score: float
    text: str


@dataclass(frozen=True)
class _SyncOutcome:
    replaced: bool
    metadata_changed: bool
    source: IndexedSource
    chunks: tuple[RagChunk, ...]
    warnings: tuple[str, ...]


def manuals_directory() -> Path:
    return _resolve_path("STUDIO_AI_MANUALS_PATH", PROJECT_ROOT / "manuals")


def index_path() -> Path:
    return _resolve_path("STUDIO_AI_RAG_INDEX_PATH", PROJECT_ROOT / "data" / "rag_index.json")


def index_display_path() -> str:
    return _display(index_path())


def validate_manual_filename(value: str) -> str:
    cleaned = value.strip()
    if cleaned != Path(cleaned).name or _MANUAL_NAME.fullmatch(cleaned) is None:
        raise ValueError("filename must be a PDF name without a directory")
    stem = Path(cleaned).stem.casefold()
    if stem in _WINDOWS_RESERVED or stem.endswith((" ", ".")):
        raise ValueError("filename is not valid on this platform")
    return cleaned


def index_chunk_count() -> int:
    with _LOCK:
        return len(_load_existing().chunks)


def search_manuals(query: str, *, limit: int = 4) -> list[ManualHit]:
    bounded = max(0, min(limit, 8))
    if bounded == 0:
        return []
    query_vector = embed_text(query)
    if not any(query_vector):
        return []
    with _LOCK:
        index = _load_existing()
    ranked: list[ManualHit] = []
    for chunk in index.chunks:
        score = _cosine(query_vector, chunk.vector)
        if score <= 0:
            continue
        ranked.append(ManualHit(source=chunk.source, page=chunk.page, score=score, text=chunk.text))
    ranked.sort(key=lambda hit: (-hit.score, hit.source, hit.page))
    return ranked[:bounded]


def ingest_manuals(filename: str | None = None) -> IngestReport:
    selected = validate_manual_filename(filename) if filename is not None else None
    with _LOCK:
        return _ingest(selected)


def embed_text(text: str) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSIONS
    for token in _TOKEN.findall(text.casefold()):
        digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % VECTOR_DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    collapsed = text.replace("\x00", " ")
    collapsed = re.sub(r"[ \t\r\f\v]+", " ", collapsed)
    collapsed = re.sub(r" *\n *", "\n", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed).strip()
    if not collapsed:
        return []
    if len(collapsed) <= size:
        return [collapsed]
    chunks: list[str] = []
    start = 0
    length = len(collapsed)
    while start < length:
        end = min(start + size, length)
        if end < length:
            break_at = collapsed.rfind("\n", start + size // 2, end)
            if break_at == -1:
                break_at = collapsed.rfind(" ", start + size // 2, end)
            if break_at != -1:
                end = break_at
        if end <= start:
            end = min(start + 1, length)
        piece = collapsed[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _ingest(filename: str | None) -> IngestReport:
    directory = manuals_directory()
    directory.mkdir(parents=True, exist_ok=True)
    sources, chunks_by_source = _loaded_maps()
    indexed: list[str] = []
    unchanged: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    dirty = not index_path().is_file()
    if filename is None:
        targets = _candidate_manuals(errors)
        full_scan = True
    else:
        targets = [find_manual(filename)]
        full_scan = False

    present: set[str] = set()
    for path in targets:
        source = _source_name(path)
        present.add(source)
        try:
            outcome = _sync_manual(path, source, sources.get(source), chunks_by_source.get(source, []))
        except ManualParseError as exc:
            if filename is not None:
                raise
            errors.append(f"{source}: {exc.reason}")
            continue
        warnings.extend(outcome.warnings)
        if outcome.replaced:
            sources[source] = outcome.source
            chunks_by_source[source] = list(outcome.chunks)
            indexed.append(source)
            dirty = True
            continue
        unchanged.append(source)
        if outcome.metadata_changed:
            sources[source] = outcome.source
            dirty = True

    removed: list[str] = []
    if full_scan:
        for name in sorted(set(sources) - present):
            removed.append(name)
            del sources[name]
            chunks_by_source.pop(name, None)
            dirty = True

    chunk_count = sum(len(items) for items in chunks_by_source.values())
    if dirty:
        rebuilt = _build_index(sources, chunks_by_source)
        _write_index(rebuilt)
        chunk_count = len(rebuilt.chunks)
    return IngestReport(
        indexed=indexed,
        unchanged=unchanged,
        removed=removed,
        warnings=warnings,
        errors=errors,
        chunk_count=chunk_count,
        index_path=_display(index_path()),
    )


def find_manual(filename: str) -> Path:
    wanted = validate_manual_filename(filename)
    directory = manuals_directory()
    if directory.is_dir():
        matches = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        for path in matches:
            if not path.is_file() or path.name.casefold() != wanted.casefold() or path.suffix.casefold() != ".pdf":
                continue
            try:
                validate_manual_filename(path.name)
            except ValueError:
                continue
            return path
    raise ManualNotFoundError(wanted)


def _candidate_manuals(errors: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for path in sorted(manuals_directory().iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() != ".pdf" or path.name.startswith("."):
            continue
        try:
            validate_manual_filename(path.name)
        except ValueError:
            errors.append(f"{path.name}: file name is not a supported PDF name")
            continue
        folded = path.name.casefold()
        if folded in seen:
            errors.append(f"manuals/{path.name}: duplicate manual name")
            continue
        seen.add(folded)
        found.append(path)
    return found


def _sync_manual(
    path: Path,
    source: str,
    previous: IndexedSource | None,
    existing: Sequence[RagChunk],
) -> _SyncOutcome:
    try:
        info = path.stat()
    except OSError as exc:
        raise ManualParseError(path.name, "PDF could not be read") from exc
    size = info.st_size
    modified = info.st_mtime_ns if info.st_mtime_ns > 0 else 0
    if size == 0:
        raise ManualParseError(path.name, "PDF is empty")
    if size > MAX_PDF_BYTES:
        raise ManualParseError(path.name, "PDF exceeds 40 MB")
    if previous is not None and previous.byte_size == size and previous.modified_ns == modified:
        return _SyncOutcome(False, False, previous, tuple(existing), ())
    digest = _sha256_file(path)
    if previous is not None and previous.content_sha256 == digest:
        refreshed = previous.model_copy(update={"byte_size": size, "modified_ns": modified})
        return _SyncOutcome(False, refreshed != previous, refreshed, tuple(existing), ())
    chunks, warnings = _extract_chunks(path, source)
    record = IndexedSource(
        source=source,
        byte_size=size,
        modified_ns=modified,
        content_sha256=digest,
        chunk_count=len(chunks),
    )
    return _SyncOutcome(True, False, record, tuple(chunks), tuple(warnings))


def _extract_chunks(path: Path, source: str) -> tuple[list[RagChunk], list[str]]:
    try:
        with path.open("rb") as handle:
            header = handle.read(5)
    except OSError as exc:
        raise ManualParseError(path.name, "PDF could not be read") from exc
    if header != b"%PDF-":
        raise ManualParseError(path.name, "file is not a PDF")
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ManualParseError(path.name, "PDF is encrypted")
        total_pages = len(reader.pages)
    except ManualParseError:
        raise
    except Exception as exc:
        logger.warning("pdf open failed file=%s error=%s", path.name, exc.__class__.__name__)
        raise ManualParseError(path.name, "PDF could not be read") from exc

    warnings: list[str] = []
    chunks: list[RagChunk] = []
    page_limit = min(total_pages, MAX_PAGES)
    if total_pages > MAX_PAGES:
        warnings.append(f"{source} indexed the first {MAX_PAGES} of {total_pages} pages")
    for page_number in range(1, page_limit + 1):
        try:
            extracted = reader.pages[page_number - 1].extract_text() or ""
        except Exception as exc:
            logger.warning(
                "pdf page failed file=%s page=%s error=%s",
                path.name,
                page_number,
                exc.__class__.__name__,
            )
            warnings.append(f"{source} page {page_number} could not be read")
            continue
        for ordinal, piece in enumerate(chunk_text(extracted), start=1):
            if len(chunks) >= MAX_CHUNKS:
                warnings.append(f"{source} reached the {MAX_CHUNKS} chunk limit")
                return chunks, warnings
            chunks.append(_make_chunk(source, page_number, ordinal, piece))
    if not chunks:
        warnings.append(f"{source} contained no extractable text")
    return chunks, warnings


def _make_chunk(source: str, page: int, ordinal: int, text: str) -> RagChunk:
    digest = hashlib.sha256(f"{source}\n{page}\n{ordinal}\n{text}".encode("utf-8")).hexdigest()
    return RagChunk(
        chunk_id=digest[:32],
        source=source,
        page=page,
        text=text,
        vector=embed_text(text),
    )


def _loaded_maps() -> tuple[dict[str, IndexedSource], dict[str, list[RagChunk]]]:
    index = _load_existing()
    sources = {item.source: item for item in index.sources}
    chunks_by_source: dict[str, list[RagChunk]] = {}
    for chunk in index.chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)
    return sources, chunks_by_source


def _load_existing() -> RagIndex:
    path = index_path()
    if not path.is_file():
        return _empty_index()
    return _read_index(path)


def _empty_index() -> RagIndex:
    return RagIndex(
        schema_version="1.0.0",
        embedding="signed_hash_v1",
        vector_dimensions=VECTOR_DIMENSIONS,
        sources=[],
        chunks=[],
    )


def _read_index(path: Path) -> RagIndex:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RagIndex.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("rag index is unreadable: %s", exc.__class__.__name__)
        raise RagIndexError() from exc


def _build_index(sources: Mapping[str, IndexedSource], chunks_by_source: Mapping[str, Sequence[RagChunk]]) -> RagIndex:
    ordered_chunks: list[RagChunk] = []
    for name in sorted(sources):
        ordered_chunks.extend(sorted(chunks_by_source.get(name, ()), key=lambda chunk: (chunk.page, chunk.chunk_id)))
    try:
        return RagIndex(
            schema_version="1.0.0",
            embedding="signed_hash_v1",
            vector_dimensions=VECTOR_DIMENSIONS,
            sources=[sources[name] for name in sorted(sources)],
            chunks=ordered_chunks,
        )
    except ValidationError as exc:
        logger.warning("rag index failed validation errors=%d", exc.error_count())
        raise RagIndexError() from exc


def _write_index(index: RagIndex) -> None:
    path = index_path()
    try:
        _atomic_write(path, index.model_dump_json(indent=2) + "\n")
    except OSError as exc:
        logger.warning("rag index could not be saved: %s", exc.__class__.__name__)
        raise RagIndexError() from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ManualParseError(path.name, "PDF could not be read") from exc
    return digest.hexdigest()


def _source_name(path: Path) -> str:
    return f"manuals/{path.name}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = math.fsum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _resolve_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    try:
        report = ingest_manuals()
    except (ManualNotFoundError, ManualParseError, RagIndexError) as exc:
        print(exc)
        return 1
    print(
        f"indexed {len(report.indexed)} unchanged {len(report.unchanged)} "
        f"removed {len(report.removed)} errors {len(report.errors)} chunks {report.chunk_count}"
    )
    for warning in report.warnings:
        print(f"warning {warning}")
    for error in report.errors:
        print(f"error {error}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
