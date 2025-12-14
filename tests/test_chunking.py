from rag.loaders import load_knowledge_base
from rag.chunking import chunk_documents

if __name__ == "__main__":
    docs = load_knowledge_base("data/knowledge_base")
    chunks = chunk_documents(docs, chunk_size=1000, overlap=150)

    print(f"Pages loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")

    for c in chunks[:2]:
        print("\n--- CHUNK SAMPLE ---")
        print("META:", c.metadata)
        print("TEXT:", c.text[:300], "...")
