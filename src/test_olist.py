from vectorstore import olist_schema_store

queries = [
    "Which table contains order price and freight?",
    "How do I identify a unique customer?",
    "Which tables contain payment information?",
    "How are orders related to order items?",
    "Which table contains product categories?",
]

for query in queries:
    print(f"\nQUERY: {query}")

    docs = olist_schema_store.similarity_search(query, k=2)

    for i, doc in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(doc.metadata)
        print(doc.page_content[:1000])