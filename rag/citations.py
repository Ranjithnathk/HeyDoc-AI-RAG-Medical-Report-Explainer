from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Citation:
    cid: int
    source: str
    page: int
    score: float
    snippet: str


def _safe_int(x, default=-1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def build_context_with_citations(
    results: List[Dict[str, Any]],
    max_snippet_chars: int = 350,
) -> tuple[str, List[Citation]]:
    """
    Build an LLM-ready context block with numbered citations and return a structured list.
    Each result is expected to have: text, score, metadata{source,page,...}
    """
    citations: List[Citation] = []
    context_parts: List[str] = []

    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {}) or {}
        source = meta.get("source", "unknown")
        page = _safe_int(meta.get("page", -1), default=-1)
        score = float(r.get("score", 0.0))

        text = (r.get("text") or "").strip()
        snippet = text[:max_snippet_chars].strip()

        citations.append(
            Citation(
                cid=i,
                source=source,
                page=page,
                score=score,
                snippet=snippet
            )
        )

        # Context entry format fed to the model
        context_parts.append(
            f"[{i}] Source: {source} (page {page})\n"
            f"{snippet}\n"
        )

    context_block = "\n".join(context_parts).strip()
    return context_block, citations


def citations_to_ui_lines(citations: List[Citation]) -> List[str]:
    """
    Simple UI-friendly citation display lines.
    """
    lines = []
    for c in citations:
        lines.append(f"[{c.cid}] {c.source} — page {c.page} (score: {c.score:.3f})")
    return lines
