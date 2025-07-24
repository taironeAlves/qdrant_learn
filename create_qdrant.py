import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings

# 1. Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL", "text-embedding-3-small")

# 2. Initialize Qdrant client and OpenAI embeddings
client = QdrantClient("http://localhost:6333")
collection_name = "livre_openai"
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# 3. Create collection if it doesn't exist
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# 4. Read database structure file
with open("estrutura_banco.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# 5. Process table blocks
table_blocks = [block.strip() for block in raw_text.split("Tabela: ") if block.strip()]
all_table_names = []
points = []

for block in table_blocks:
    lines = block.splitlines()
    table_name = lines[0].strip()
    all_table_names.append(table_name)

    table_structure = "\n".join(lines[1:]).strip()
    content = f"Table: {table_name}\n{table_structure}"

    vector = embeddings.embed_query(content)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "page_content": content,
            "table_name": table_name
        }
    )
    points.append(point)

# 6. Upload to Qdrant
client.upsert(collection_name=collection_name, points=points)

print(f"\n✅ {len(points)} tables indexed in collection '{collection_name}' successfully.")
