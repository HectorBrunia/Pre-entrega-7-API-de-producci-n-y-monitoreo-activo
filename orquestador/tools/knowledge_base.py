"""
Base de conocimiento de TechCorp: RAG híbrido (BM25 + Pinecone) de las pre-entregas 3/4.

- Si el índice/namespace de Pinecone ya tiene vectores, solo se conecta (no reingesta).
- Si no hay PINECONE_API_KEY o Pinecone falla, degrada a BM25 local: el orquestador
  sigue funcionando (resiliencia) y se deja constancia en el log.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
INDEX_NAME = os.getenv("PINECONE_INDEX", "techcorp-rag-hibrido")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "orquestador-pe6")
TOP_K = 4


def cargar_chunks():
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader(
        str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(loader.load())
    for i, c in enumerate(chunks):
        c.metadata["source"] = os.path.basename(c.metadata["source"])
        c.metadata["chunk_id"] = i
    return chunks


def _retriever_pinecone(chunks):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace=NAMESPACE)
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    if stats["namespaces"].get(NAMESPACE, {}).get("vector_count", 0) == 0:
        logger.info("Namespace '%s' vacío: ingestando %d chunks", NAMESPACE, len(chunks))
        vectorstore.add_documents(chunks)
    else:
        logger.info("Namespace '%s' ya tiene datos: no se reingesta", NAMESPACE)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K, "namespace": NAMESPACE})


_LOCK = threading.Lock()


def obtener_retriever():
    """Thread-safe: con 5 tareas concurrentes, la herramienta corre en varios threads a la vez
    y sin el lock cada uno construiría (y cargaría en memoria) su propio índice."""
    with _LOCK:
        return _construir_retriever()


@lru_cache(maxsize=1)
def _construir_retriever():
    """Construye el retriever una sola vez (lazy) y lo reutiliza."""
    from langchain_community.retrievers import BM25Retriever

    chunks = cargar_chunks()
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K

    if not os.getenv("PINECONE_API_KEY"):
        logger.warning("Sin PINECONE_API_KEY: usando solo BM25 local")
        return bm25
    try:
        from langchain_classic.retrievers import EnsembleRetriever

        vectorial = _retriever_pinecone(chunks)
        return EnsembleRetriever(retrievers=[bm25, vectorial], weights=[0.5, 0.5])
    except Exception as e:  # degradación controlada
        logger.warning("Pinecone no disponible (%s): usando solo BM25 local", e)
        return bm25


def buscar(query: str, k: int = TOP_K) -> list[dict]:
    docs = obtener_retriever().invoke(query)[:k]
    return [{"contenido": d.page_content, "fuente": d.metadata.get("source", "?")} for d in docs]
