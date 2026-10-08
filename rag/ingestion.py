from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


def load_documents(data_dir: str):
    documents = []

    base_path = Path(data_dir)

    for path in base_path.rglob("*.md"):
        content = path.read_text(encoding="utf-8")

        relative_path = path.relative_to(base_path)

        topic = relative_path.parts[0]

        documents.append({
            "content": content,
            "topic": topic,
            "filename": path.name,
            "path": str(relative_path)
        })

    return documents


def create_chunks(documents):
    all_chunks = []

    for document in documents:

        chunks = splitter.create_documents(
            [document["content"]],
            metadatas=[{
                "topic": document["topic"],
                "filename": document["filename"],
                "path": document["path"]
            }]
        )

        all_chunks.extend(chunks)

    return all_chunks