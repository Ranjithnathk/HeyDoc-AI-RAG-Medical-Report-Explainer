from rag.retriever import retrieve_top_k
from rag.citations import build_context_with_citations, citations_to_ui_lines

if __name__ == "__main__":
    query = "What does ground-glass opacity mean in a radiology report?"

    results = retrieve_top_k(query)
    context, citations = build_context_with_citations(results)

    print("=== CONTEXT BLOCK (sent to LLM) ===")
    print(context[:1200], "...\n")

    print("=== CITATIONS (UI) ===")
    for line in citations_to_ui_lines(citations):
        print(line)
