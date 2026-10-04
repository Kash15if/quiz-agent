from langchain_core.tools import tool


@tool
def read_document(path: str = "data/document.md") -> str:
    """
    Read the Markdown document containing the quiz source material.
    """

    with open(path, "r", encoding="utf-8") as file:
        return file.read()