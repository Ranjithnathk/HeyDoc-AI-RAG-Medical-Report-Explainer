import json
import time
from pathlib import Path

from rag.retriever import retrieve_top_k
from rag.citations import build_context_with_citations, citations_to_ui_lines
from app.prompts import SYSTEM_BASE, qa_prompt
from app.generate import generate_text
from app.guards import enforce_disclaimer


def has_citation_markers(text: str) -> bool:
    """Heuristic: detect any [1]..[10] marker in the answer."""
    if not text:
        return False
    for i in range(1, 11):
        if f"[{i}]" in text:
            return True
    return False


def main():
    eval_path = Path("eval/eval_set.json")
    out_path = Path("eval/results.json")

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    queries = data["queries"]

    results = {
        "project": data.get("project"),
        "version": data.get("version"),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_queries": len(queries),
        "items": [],
        "summary": {}
    }

    total_latency = 0.0
    citations_yes = 0
    evidence_yes = 0
    evidence_and_cited = 0

    # Retrieval settings (keep consistent with app defaults)
    TOP_K = 12
    MIN_SCORE = 0.50
    FINAL_TOP_K = 6

    for q in queries:
        qid = q["id"]
        qtype = q.get("type")
        question = q["question"]

        t0 = time.time()

        # Retrieve evidence
        retrieved = retrieve_top_k(question, top_k=TOP_K, min_score=MIN_SCORE, final_top_k=FINAL_TOP_K)
        
        # If no evidence, do NOT generate and do NOT cite (more defensible)
        if len(retrieved) == 0:
            latency = round(time.time() - t0, 3)
            total_latency += latency

            item = {"id": qid, "type": qtype, "question": question, "latency_sec": latency,
                    "retrieved_chunks_count": 0, "citations_present": False, "citations_ui": [],
                    "top_sources": [], "answer": "I don't know based on the provided sources."}
            results["items"].append(item)
            print(f"{qid}: latency={latency}s, chunks=0, citations=False (no evidence)")
            continue

        evidence_yes += 1
        
        # Build citation context
        context_block, citations = build_context_with_citations(retrieved)

        # Generate answer using Q&A prompt (uses evidence context + citations)
        user_prompt = qa_prompt(question, report_text="", evidence_context=context_block)
        answer = generate_text(SYSTEM_BASE, user_prompt)
        answer = enforce_disclaimer(answer)

        t1 = time.time()
        latency = round(t1 - t0, 3)
        total_latency += latency

        cite_ok = has_citation_markers(answer)
        if cite_ok:
            citations_yes += 1
            evidence_and_cited += 1

        item = {
            "id": qid,
            "type": qtype,
            "question": question,
            "latency_sec": latency,
            "retrieved_chunks_count": len(retrieved),
            "citations_present": cite_ok,
            "citations_ui": citations_to_ui_lines(citations),
            "top_sources": [
                {"source": c.source, "page": c.page, "score": round(c.score, 4)} for c in citations
            ],
            "answer": answer
        }

        results["items"].append(item)
        print(f"{qid}: latency={latency}s, chunks={len(retrieved)}, citations={cite_ok}")

    n = len(queries)
    avg_latency = round(total_latency / max(n, 1), 3)
    citation_coverage = round(citations_yes / max(n, 1), 3)
    evidence_rate = round(evidence_yes / max(n, 1), 3)
    citation_when_evidence = round(evidence_and_cited / max(evidence_yes, 1), 3)
    results["summary"] = {
        "avg_latency_sec": avg_latency,
        "citation_coverage": citation_coverage,
        "evidence_rate": evidence_rate,
        "citation_coverage_given_evidence": citation_when_evidence,
        "retrieval_settings": {
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "final_top_k": FINAL_TOP_K,
        },
        "notes": (
            "citation_coverage detects [1]..[10] markers in answers. "
            "If retrieval returns 0 chunks, the script returns an 'I don't know' answer without citations."
        ),
    }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
