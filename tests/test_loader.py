from rag.loaders import load_knowledge_base

if __name__ == "__main__":
    docs = load_knowledge_base("data/knowledge_base")
    print(f"Loaded page-chunks: {len(docs)}")

    # Show 2 samples
    for d in docs[:2]:
        print("\n--- SAMPLE ---")
        print("META:", d.metadata)
        print("TEXT:", d.text[:300], "...")
