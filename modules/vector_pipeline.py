"""ChromaDB-backed in-memory vector RAG (Pipeline A)."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore


def build_vector_rag(documents, embed_model, chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Index raw text into a volatile in-memory Chroma collection.

    SentenceSplitter (vs SimpleNodeParser) respects sentence boundaries,
    which often improves retrieval quality on spec-style prose.
    """
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)

    # Ephemeral client: no disk persistence — matches zero-local-ops goal
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("rag_arena_vector")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index.as_query_engine(similarity_top_k=3)
