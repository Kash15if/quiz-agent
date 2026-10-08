from rag.vector_store import vector_store


def main():

    query = "How does inheritance work in Java?"

    print("Query:", query)
    print("\nSearching...\n")

    results = vector_store.similarity_search(
        query,
        k=5
    )

    for i, document in enumerate(results):

        print("=" * 80)
        print(f"RESULT {i + 1}")
        print("=" * 80)

        print("Metadata:")
        print(document.metadata)

        print("\nContent:")
        print(document.page_content)


if __name__ == "__main__":
    main()