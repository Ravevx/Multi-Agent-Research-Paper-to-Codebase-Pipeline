from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import LMSTUDIOURL


def build_rag(chunks: list):
    """Build a FAISS vector store from paper chunks. Call once per paper."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(chunks)

    embeddings = OpenAIEmbeddings(
        base_url=LMSTUDIOURL,
        api_key="lm-studio",
        model="text-embedding-nomic-embed-text-v1.5",
        check_embedding_ctx_length=False
    )

    store = FAISS.from_documents(docs, embeddings)
    print(f"RAG store built: {len(docs)} chunks indexed")
    return store


def retrieve(store, query: str, k: int = 5) -> str:
    """Retrieve the k most relevant chunks for a query."""
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)
