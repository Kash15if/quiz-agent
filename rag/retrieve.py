from langchain_ollama import ChatOllama

from rag.vector_store import vector_store


llm = ChatOllama(
    model="phi3",
    temperature=0
)


ANSWER_SYSTEM_PROMPT = """
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Use only information found in the context.
2. If the context does not contain the answer, reply exactly:
   "I don't know based on the provided documents."
3. Do not use outside knowledge.
4. Do not invent facts.
5. Cite the sources you used with their bracketed numbers, e.g. [1].
6. Keep the answer clear, concise and grounded.

The context is the only source of truth.
"""


def build_context(results):
    blocks = []

    for i, document in enumerate(results, start=1):

        metadata = document.metadata

        heading_path = " > ".join(
            metadata[key]
            for key in ("h1", "h2", "h3", "h4")
            if metadata.get(key)
        )

        header = f"[{i}] {metadata.get('path', 'unknown')}"

        if heading_path:
            header += f" | {heading_path}"

        blocks.append(f"{header}\n{document.page_content}")

    return "\n\n".join(blocks)


def answer_question(query, k=5):

    results = vector_store.similarity_search(query, k=k)

    if not results:
        return "I don't know based on the provided documents.", []

    context = build_context(results)

    prompt = f"""
{ANSWER_SYSTEM_PROMPT}

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, results


def main():

    query = "How does inheritance work in Java?"

    print("Query:", query)
    print("\nSearching...\n")

    answer, results = answer_question(query, k=5)

    print("Answer:\n")
    print(answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for i, document in enumerate(results, start=1):
        print(f"[{i}] {document.metadata}")


if __name__ == "__main__":
    main()
