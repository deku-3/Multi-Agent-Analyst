from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.vectorstore import olist_schema_store


# Load the same documents already stored in Chroma
data = olist_schema_store.get(
    include=["documents", "metadatas"]
)

DOCUMENTS = [
    Document(
        page_content=text,
        metadata=metadata or {},
    )
    for text, metadata in zip(
        data["documents"],
        data["metadatas"],
    )
]

BM25 = BM25Okapi(
    [doc.page_content.lower().split() for doc in DOCUMENTS]
)


def hybrid_retrieve(query: str, k: int = 5):
    # Dense retrieval
    dense_docs = olist_schema_store.similarity_search(
        query,
        k=k,
    )

    # BM25 retrieval
    scores = BM25.get_scores(
        query.lower().split()
    )

    bm25_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    bm25_docs = [
        DOCUMENTS[i]
        for i in bm25_indices
    ]

    # RRF
    rrf_scores = {}
    doc_map = {}

    for ranked_docs in [dense_docs, bm25_docs]:
        for rank, doc in enumerate(ranked_docs, start=1):

            key = (
                doc.metadata.get("dataset", "olist"),
                doc.metadata.get("doc_type", ""),
                doc.metadata.get("table", ""),
            )

            rrf_scores[key] = (
                rrf_scores.get(key, 0)
                + 1 / (60 + rank)
            )

            doc_map[key] = doc

    ranked = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    return [
        doc_map[key]
        for key in ranked[:k]
    ]