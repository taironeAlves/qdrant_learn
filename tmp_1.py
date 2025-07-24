from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Verificar se a API key está configurada
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")

def initialize_qdrant_connection():
    """Inicializa conexão com Qdrant com tratamento de erros"""
    try:
        client = QdrantClient("http://localhost:6333")
        # Testar conexão
        client.get_collections()
        print("✅ Conexão com Qdrant estabelecida")
        return client
    except Exception as e:
        print(f"❌ Erro ao conectar com Qdrant: {e}")
        print("Verifique se o Qdrant está rodando em localhost:6333")
        return None

def initialize_vector_store(client, collection_name):
    """Inicializa o vector store com verificações"""
    try:
        # Verificar se a coleção existe
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if not collection_exists:
            print(f"⚠️  Coleção '{collection_name}' não existe.")
            print("Criando coleção...")
            
            # Criar coleção se não existir
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Tamanho do embedding text-embedding-3-small
                    distance=models.Distance.COSINE
                )
            )
            print(f"✅ Coleção '{collection_name}' criada")
        
        # Inicializar embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
        )
        
        # Inicializar vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,  # Note: 'embeddings', não 'embedding'
        )
        
        print("✅ Vector store inicializado")
        return vector_store
        
    except Exception as e:
        print(f"❌ Erro ao inicializar vector store: {e}")
        return None

def extract_tables_from_input(user_input):
    """Extrai tabelas mencionadas no input do usuário"""
    known_tables = ["chat_list", "channel", "contact"]
    mentioned_tables = [
        table for table in known_tables 
        if table.lower() in user_input.lower()
    ]
    print(f"🔍 Tabelas encontradas no input: {mentioned_tables}")
    return mentioned_tables

def search_with_table_filter(vector_store, query):
    """Realiza busca com filtro de tabelas"""
    try:
        # Extrair tabelas mencionadas
        tables = extract_tables_from_input(query)
       
        # Configurar filtro de metadados
        metadata_filter = None
        if tables:
            # Opção 1: Usando MatchAny com 'any'
            try:
                metadata_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="table_name",
                            match=models.MatchAny(any=tables)
                        )
                    ]
                )
            except Exception as e:
                print(f"Tentando sintaxe alternativa do filtro...")
                # Opção 2: Usando múltiplos MatchValue em OR
                conditions = [
                    models.FieldCondition(
                        key="table_name",
                        match=models.MatchValue(value=table)
                    ) for table in tables
                ]
                
                metadata_filter = models.Filter(
                    should=conditions  # OR entre as condições
                )
            
            print(f"🔧 Aplicando filtro para tabelas: {tables}")
        else:
            print("🔧 Sem filtro de tabela aplicado")
       
        # Realizar busca vetorial
        results = vector_store.similarity_search(
            query,
            k=5,
            filter=metadata_filter
        )
       
        print(f"📊 Encontrados {len(results)} resultados")
        return results
        
    except Exception as e:
        print(f"❌ Erro durante a busca: {e}")
        return []

def check_collection_data(client, collection_name):
    """Verifica se há dados na coleção"""
    try:
        info = client.get_collection(collection_name)
        point_count = info.points_count
        print(f"📈 Coleção '{collection_name}' tem {point_count} pontos")
        
        if point_count == 0:
            print("⚠️  A coleção está vazia! Você precisa adicionar documentos primeiro.")
            return False
        return True
        
    except Exception as e:
        print(f"❌ Erro ao verificar coleção: {e}")
        return False

def main():
    """Função principal com tratamento completo de erros"""
    collection_name = "livre_openai"
    
    # 1. Inicializar conexão com Qdrant
    client = initialize_qdrant_connection()
    if not client:
        return
    
    # 2. Verificar se há dados na coleção
    if not check_collection_data(client, collection_name):
        return
    
    # 3. Inicializar vector store
    vector_store = initialize_vector_store(client, collection_name)
    if not vector_store:
        return
    
    # 4. Realizar busca
    # user_query = "Mostre as colunas da tabela chat_list e contact"
    #user_query = "Mostre as colunas dados de usuarios"
    user_query = "mostre as chaves estrangeiras da tabela channel"
    # user_query = "Mostre alguma coisa fora da caixa"
    print(f"🔍 Buscando por: '{user_query}'")
    
    results = search_with_table_filter(vector_store, user_query)
    
    # 5. Processar e exibir resultados
    if results:
        print("\n📋 RESULTADOS:")
        print("=" * 50)
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Tabela: {doc.metadata.get('table_name', 'N/A')}")
            print(f"   Conteúdo: {doc.page_content}")
            print(f"   Metadados: {doc.metadata}")
    else:
        print("❌ Nenhum resultado encontrado")

if __name__ == "__main__":
    main()