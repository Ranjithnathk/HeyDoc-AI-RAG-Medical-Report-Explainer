from rag.retriever import retrieve_top_k

if __name__ == "__main__":
    query = "What does ground-glass opacity mean in a radiology report?"

    results = retrieve_top_k(query, top_k=5)

    print(f"Retrieved {len(results)} filtered chunks\n")

    for i, r in enumerate(results, 1):
        print(f"--- RESULT {i} ---")
        print("Score:", round(r["score"], 4))
        print("Source:", r["metadata"].get("source"))
        print("Page:", r["metadata"].get("page"))
        print("Text:", r["text"][:250], "...\n")
