from __future__ import annotations

from typing import List, Dict, Any, Tuple


def dedupe_by_source_page(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the best-scoring chunk per (source, page).
    Prevents the context from being dominated by one page.
    """
    best: dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in results:
        meta = r.get("metadata", {})
        key = (meta.get("source", "unknown"), int(meta.get("page", -1)))
        if key not in best or r.get("score", 0) > best[key].get("score", 0):
            best[key] = r
    # preserve sorting by score descending
    return sorted(best.values(), key=lambda x: x.get("score", 0), reverse=True)


def filter_by_threshold(
    results: List[Dict[str, Any]],
    min_score: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Remove weak matches. Score ranges vary; 0.50 is a good starting point.
    """
    return [r for r in results if r.get("score", 0.0) >= min_score]


def trim_to_max_chars(
    results: List[Dict[str, Any]],
    max_context_chars: int = 4500,
    per_chunk_char_cap: int = 900,
) -> List[Dict[str, Any]]:
    """
    Keep adding chunks until we hit max_context_chars.
    Also cap each chunk length to keep context readable.
    """
    trimmed = []
    total = 0

    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue

        text = text[:per_chunk_char_cap]
        new_total = total + len(text)
        if new_total > max_context_chars:
            break

        r2 = dict(r)
        r2["text"] = text
        trimmed.append(r2)
        total = new_total

    return trimmed


def rank_and_filter(
    raw_results: List[Dict[str, Any]],
    min_score: float = 0.50,
    max_context_chars: int = 4500,
    per_chunk_char_cap: int = 900,
    final_top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Full ranking/filtering pipeline:
    1) score threshold
    2) dedupe by (source,page)
    3) sort by score
    4) limit count
    5) trim by total character budget
    """
    step1 = filter_by_threshold(raw_results, min_score=min_score)
    step2 = dedupe_by_source_page(step1)
    step3 = step2[:final_top_k]
    step4 = trim_to_max_chars(
        step3,
        max_context_chars=max_context_chars,
        per_chunk_char_cap=per_chunk_char_cap
    )
    return step4
