from qdrant_client import QdrantClient
from openai import OpenAI
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from dotenv import load_dotenv
import os

load_dotenv()


openai_client = OpenAI()
qdrant_client = QdrantClient("http://localhost:6333")
collection_name = "livre_openai"


def extract_tables_from_input(user_input):
    try:
        with open("lista_de_tabelas.txt", "r") as file:
            known_tables = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print("Arquivo 'lista_de_tabelas.txt' não encontrado.")
        known_tables = []

    mentioned_tables = [
        table for table in known_tables if table.lower() in user_input.lower()
    ]
    return mentioned_tables


def search_with_table_filter(query):
    tables = extract_tables_from_input(query)

    if tables:
        metadata_filter = Filter(
            must=[FieldCondition(key="table_name", match=MatchAny(any=tables))]
        )
    else:
        metadata_filter = None

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=embedding,
        query_filter=metadata_filter,
        limit=5,
    )

    return results


user_query = "Mostre alguma coisa fora da caixa"

embedding = (
    openai_client.embeddings.create(model="text-embedding-3-small", input=user_query)
    .data[0]
    .embedding
)


result = search_with_table_filter(user_query)

for r in result.points:
    context = "\n\n".join([r.payload["page_content"] for r in result.points])

resp = f"Contexto:\n{context}\n\nPergunta:\n{user_query}"

response = openai_client.responses.create(
    prompt={
        "id": "pmpt_687b097121dc8197a7c2de012e58227a00fb5a28ba6fcda4",
    },
    input=resp,
)

print(response.output_text)

# if result.points:
#     print("\nResultados encontrados:")
#     for point in result.points:
#         # print(f"Tabela: {point.payload.get('table_name')}")
#         print(point.payload.get("page_content"))
#         print("\n\n")
# else:
#     print("Nenhum resultado encontrado.")
