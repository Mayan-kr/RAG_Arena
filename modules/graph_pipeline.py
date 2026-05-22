"""Neo4j Property Graph RAG (Pipeline B)."""

from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


def build_graph_rag(
    documents,
    llm,
    embed_model,
    url: str,
    username: str,
    password: str,
    database: str = "neo4j",
):
    """
    Extract entity-relation triplets via LLM and persist in Neo4j Aura.

    Uses Neo4jPropertyGraphStore with refresh_schema=False so startup does not
    require apoc.meta.data() — which is restricted or unavailable on many
    Aura configurations. PropertyGraphIndex is the supported API for this.
    """
    graph_store = Neo4jPropertyGraphStore(
        url=url,
        username=username,
        password=password,
        database=database,
        refresh_schema=False,
    )

    try:
        existing = graph_store.structured_query("MATCH (n) RETURN count(n) AS count")
        has_data = existing and existing[0].get("count", 0) > 0
    except Exception:
        has_data = False

    if has_data:
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm,
            embed_model=embed_model,
        )
    else:
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            llm=llm,
            embed_model=embed_model,
            show_progress=True,
        )
    return index.as_query_engine()

