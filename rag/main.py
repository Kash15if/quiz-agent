from rag.ingestion import load_documents, create_chunks
from rag.vector_store import vector_store


def main():

    documents = load_documents("data")

    print("Documents:", len(documents))

    chunks = create_chunks(documents)

    print("Chunks:", len(chunks))

    print("Clearing existing collection...")

    vector_store.delete_collection()
    vector_store.create_collection()

    print("Adding chunks to PostgreSQL...")

    vector_store.add_documents(chunks)

    print("Done!")


if __name__ == "__main__":
    main()