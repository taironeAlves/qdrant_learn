from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Qdrant
client = QdrantClient("http://localhost:6333")

# Check Qdrant connection
try:
    client.get_collections()
    print("Conexão com Qdrant estabelecida com sucesso.")
except Exception as e:
    print(f"Erro ao conectar com Qdrant: {e}")
    exit(1)

# Define embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Define vector store
collection_name = "livre_openai"

# Create collection if it doesn't exist
try:
    client.get_collection(collection_name)
    print(f"Coleção '{collection_name}' já existe.")
except Exception as e:
    print(f"Coleção '{collection_name}' não encontrada. Criando...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,  
            distance=models.Distance.COSINE,
        ),
    )

vector_store = QdrantVectorStore(
    embedding=embeddings,
    collection_name=collection_name,
    client=client,
)

def extract_tables_from_input(user_input):
    # Lista de tabelas conhecidas
    known_tables = ["chat_list", "channel", "contact"]
    # Encontrar tabelas mencionadas no input
    mentioned_tables = [
        table for table in known_tables if table.lower() in user_input.lower()
    ]
    return mentioned_tables

def search_with_table_filter(query):
    # Extrair tabelas mencionadas
    tables = extract_tables_from_input(query)
   
    # Configurar filtro de metadados usando o modelo do Qdrant
    if tables:
        metadata_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="table_name",
                    match=models.MatchAny(any=tables)  
                )
            ]
        )
    else:
        metadata_filter = None
   
    # Realizar busca vetorial
    results = vector_store.similarity_search(
        query,
        k=5,  # Número de resultados
        filter=metadata_filter  # Filtro opcional
    )
   
    return results

# Exemplo de busca
user_query = "Mostre as colunas da tabela chat_list e contact"
print(f"\nBuscando por: '{user_query}'")
results = search_with_table_filter(user_query)

# Processar resultados
if results:
    print("\nResultados encontrados:")
    for doc in results:
        print(f"Tabela: {doc.metadata.get('table_name')}")
        print(f"Conteúdo: {doc.page_content}")
        print("\n\n")
        print("-" * 50)
else:
    print("Nenhum resultado encontrado.")