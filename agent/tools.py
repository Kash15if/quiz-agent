from rag.vector_store import vector_store


def retrieve_context(query: str, k: int = 15) -> str:
    """
    Retrieve the most relevant chunks from the vector store
    for a given query and return them as one context string.
    """

    results = vector_store.similarity_search(query, k=k)

    blocks = [
        document.page_content
        for document in results
    ]

    return "\n\n".join(blocks)
