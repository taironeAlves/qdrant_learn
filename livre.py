from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import re

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Qdrant
client = QdrantClient("http://localhost:6333")

# Define embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

# Define vector store
collection_name = "livre_openai"
vector_store = QdrantVectorStore(
    embedding=embeddings,
    collection_name=collection_name,
    client=client,
)

# Query
query = "Mostre as colunas da tabela chat_list e contact"

# Detect table names
table_matches = re.findall(r"tabela[s]? (\w+)|table[s]? (\w+)", query.lower())
table_names = [name[0] or name[1] for name in table_matches]
table_names = list(set(table_names))  # remove duplicates

# Apply filter only if tables are mentioned
if table_names:
    table_filter = Filter(
        must=[
            FieldCondition(
                key="table_name",
                match=MatchAny(any=table_names)
            )
        ]
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10, "filter": table_filter})
else:
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Search
docs = retriever.invoke(query)

# Output
print(f"🔍 Tabelas detectadas: {table_names if table_names else 'Nenhuma'}")
print(f"📄 Resultados: {len(docs)} documentos\n")
for doc in docs:
    print(f"\n---\n{doc.page_content}")
