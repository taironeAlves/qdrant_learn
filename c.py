from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Inicialização
client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "livre_openai"

# Consulta do usuário
query = "Mostre as colunas da table chat_list"

# 1. Embedding da pergunta
embedding = (
    client.embeddings.create(model="text-embedding-3-small", input=query)
    .data[0]
    .embedding
)

# 2. Busca no Qdrant
resultados = qdrant.query_points(
    collection_name=collection_name,
    query=embedding,
    limit=10,
).points

for r in resultados:

    # 3. Montar contexto
    context = "\n\n".join([r.payload["page_content"] for r in resultados])

# 4. Enviar para prompt pré-definido via ID
resp = f"Contexto:\n{context}\n\nPergunta:\n{query}"
response = client.responses.create(
    prompt={
        "id": "pmpt_687b097121dc8197a7c2de012e58227a00fb5a28ba6fcda4",
        "variables": {
            "context": context,
        },
    },
    input=resp,
)

print(response.output_text)
