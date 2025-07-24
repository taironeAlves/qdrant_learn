# https://www.youtube.com/watch?v=KRiFMFVPL1Q
# import client qdrant
from qdrant_client import QdrantClient

# cria a collection
from qdrant_client.http.models import Distance, VectorParams

# Criar nossos vetoes
from qdrant_client.http.models import PointStruct

# Filtrar por vetores
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Pretty Print
from pprint import pprint

client = QdrantClient(url="http://localhost:6333")

# Crio a coleção dentro do qdrant
#client.create_collection(
#     collection_name="test_collection",
#     vectors_config=VectorParams(
#         size=4,
#         distance=Distance.DOT
#     )
# )

operation_info = client.upsert(
    collection_name="test_collection",
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
)   
pprint(operation_info)

search_result = client.query_points(
    collection_name="test_collection",
    query=[0.2, 0.1, 0.9, 0.7],
     query_filter=Filter(
        must=[FieldCondition(key="city", match=MatchValue(value="London"))]
    ),
    with_payload=True,
    limit=3
).points

pprint(search_result)
