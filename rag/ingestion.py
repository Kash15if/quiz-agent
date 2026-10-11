from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=False
)

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

        base_metadata = {
            "topic": document["topic"],
            "filename": document["filename"],
            "path": document["path"]
        }

        sections = header_splitter.split_text(document["content"])

        for section in sections:

            metadata = {**base_metadata, **section.metadata}

            chunks = splitter.create_documents(
                [section.page_content],
                metadatas=[metadata]
            )

            all_chunks.extend(chunks)

    return all_chunks