import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def main():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "heydocai-medkb")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist (dimension must match embedding model)
    # text-embedding-3-small => 1536 dims
    dims = 1536

    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dims,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Pinecone index already exists: {index_name}")

    index = pc.Index(index_name)

    # Upsert a tiny vector just to prove it works
    test_vector = [0.0] * dims
    test_vector[0] = 0.001  # any non-zero value works
    index.upsert(vectors=[{
        "id": "smoke-test-1",
        "values": test_vector,
        "metadata": {"source": "smoke_test", "text": "hello pinecone"}
    }])

    res = index.query(vector=test_vector, top_k=1, include_metadata=True)
    match = res["matches"][0] if res.get("matches") else None

    if match:
        print("Query returned a match:")
        print(match)
    else:
        raise RuntimeError("Query returned no matches. Something is wrong.")

    print("Pinecone smoke test PASSED.")

if __name__ == "__main__":
    main()
