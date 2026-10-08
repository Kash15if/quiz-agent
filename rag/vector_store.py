from langchain_postgres import PGVector

from rag.embeddings import embeddings


DATABASE_URL = (
    "postgresql+psycopg://postgres:@localhost:5432/rag_db"
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="quiz_documents",
    connection=DATABASE_URL,
)