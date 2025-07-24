# Armazena e consulta perguntas e respostas no Qdrant (claude)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import openai
import hashlib
import uuid
from datetime import datetime
import asyncio
import os

class QdrantOpenAIQAService:
    def __init__(self, openai_api_key=None, qdrant_host="localhost", qdrant_port=6333):
        """
        Inicializa o serviço de Q&A com Qdrant e OpenAI embeddings
        
        Args:
            openai_api_key: Sua chave da API OpenAI
            qdrant_host: Host do Qdrant (padrão: localhost)
            qdrant_port: Porta do Qdrant (padrão: 6333)
        """
        # Setup OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Setup Qdrant
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "qa_pairs_openai"
        
        # Configurações do embedding
        self.embedding_model = "text-embedding-3-small"  # Mais barato: $0.02/1M tokens
        self.embedding_dimension = 1536  # Dimensão padrão do text-embedding-3-small
        
        # Setup da collection
        self.setup_collection()
    
    def setup_collection(self):
        """Cria collection no Qdrant se não existir"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' já existe")
        except Exception:
            print(f"🔧 Criando collection '{self.collection_name}'...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{self.collection_name}' criada com sucesso!")
    
    async def encode_question(self, question):
        """
        Converte pergunta em embedding usando OpenAI
        
        Args:
            question: Pergunta em texto
            
        Returns:
            List: Vector embedding da pergunta
        """
        try:
            response = await openai.Embedding.acreate(
                model=self.embedding_model,
                input=question.strip(),
                encoding_format="float"
            )
            
            embedding = response['data'][0]['embedding']
            print(f"🔤 Embedding gerado: {len(embedding)} dimensões")
            return embedding
            
        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
            raise
    
    async def get_cached_answer(self, question, similarity_threshold=0.85, limit=3):
        """
        Busca resposta similar no cache do Qdrant
        
        Args:
            question: Pergunta do usuário
            similarity_threshold: Threshold mínimo de similaridade (0.0-1.0)
            limit: Número máximo de resultados
            
        Returns:
            Dict ou None: Dados da resposta em cache ou None se não encontrar
        """
        try:
            # Gera embedding da pergunta
            query_vector = await self.encode_question(question)
            
            # Busca no Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=similarity_threshold,
                with_payload=True
            )
            
            if results and results[0].score >= similarity_threshold:
                best_match = results[0]
                
                # Incrementa contador de uso
                await self.increment_usage(best_match.id)
                
                print(f"✅ Cache HIT! Score: {best_match.score:.4f}")
                print(f"📝 Pergunta original: {best_match.payload['question'][:100]}...")
                
                return {
                    'answer': best_match.payload['answer'],
                    'similarity': best_match.score,
                    'cached': True,
                    'original_question': best_match.payload['question'],
                    'usage_count': best_match.payload.get('usage_count', 1),
                    'point_id': best_match.id
                }
            
            print(f"❌ Cache MISS - Melhor score: {results[0].score:.4f if results else 'N/A'}")
            return None
            
        except Exception as e:
            print(f"❌ Erro na busca cache: {e}")
            return None
    
    async def store_qa_pair(self, question, answer, user_id, confidence_score=1.0):
        """
        Armazena pergunta e resposta no Qdrant
        
        Args:
            question: Pergunta original
            answer: Resposta da IA
            user_id: ID do usuário
            confidence_score: Score de confiança (0.0-1.0)
            
        Returns:
            str: ID do ponto armazenado
        """
        try:
            point_id = str(uuid.uuid4())
            
            # Gera embedding da pergunta
            vector = await self.encode_question(question)
            
            # Cria ponto com payload completo
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "question": question,
                    "answer": answer,
                    "user_id": user_id,
                    "confidence_score": confidence_score,
                    "usage_count": 1,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "validation_status": "validated",
                    "model_used": "gpt-4.1-nano-2025-04-14",
                    "embedding_model": self.embedding_model
                }
            )
            
            # Armazena no Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            print(f"💾 Q&A armazenado com ID: {point_id}")
            return point_id
            
        except Exception as e:
            print(f"❌ Erro ao armazenar: {e}")
            raise
    
    async def increment_usage(self, point_id):
        """
        Incrementa contador de uso de uma resposta
        
        Args:
            point_id: ID do ponto no Qdrant
        """
        try:
            # Pega dados atuais
            point = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True
            )[0]
            
            # Incrementa usage_count
            current_count = point.payload.get('usage_count', 1)
            new_payload = point.payload.copy()
            new_payload['usage_count'] = current_count + 1
            new_payload['updated_at'] = datetime.now().isoformat()
            
            # Atualiza no Qdrant
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload=new_payload,
                points=[point_id]
            )
            
            print(f"📊 Usage count atualizado: {current_count} → {current_count + 1}")
            
        except Exception as e:
            print(f"❌ Erro ao incrementar usage: {e}")
    
    async def call_openai(self, question, prompt_id=None):
        """
        Chama a API da OpenAI para gerar resposta
        
        Args:
            question: Pergunta do usuário
            prompt_id: ID do assistant/prompt personalizado
            
        Returns:
            str: Resposta da OpenAI
        """
        try:
            # Se você tem um Assistant/GPT personalizado, use assim:
            if prompt_id:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4.1-nano-2025-04-14",
                    messages=[
                        {"role": "system", "content": f"Use prompt ID: {prompt_id}"},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=2500,
                    temperature=0.7
                )
            else:
                # Prompt padrão
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4.1-nano-2025-04-14",
                    messages=[
                        {"role": "system", "content": "Você é um assistente útil que responde perguntas de forma clara e precisa."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=2500,
                    temperature=0.7
                )
            
            answer = response.choices[0].message.content
            print(f"🤖 Resposta OpenAI gerada: {len(answer)} caracteres")
            return answer
            
        except Exception as e:
            print(f"❌ Erro na chamada OpenAI: {e}")
            raise
    
    async def answer_question(self, question, user_id, prompt_id=None, similarity_threshold=0.85):
        """
        Pipeline completo: busca cache → OpenAI → armazena
        
        Args:
            question: Pergunta do usuário
            user_id: ID do usuário
            prompt_id: ID do prompt/assistant personalizado
            similarity_threshold: Threshold de similaridade para cache
            
        Returns:
            Dict: Resposta com metadados
        """
        print(f"\n🔍 Processando pergunta de {user_id}: {question[:100]}...")
        
        # 1. Tenta buscar em cache primeiro
        cached = await self.get_cached_answer(question, similarity_threshold)
        if cached:
            return {
                'answer': cached['answer'],
                'source': 'cache',
                'similarity': cached['similarity'],
                'original_question': cached['original_question'],
                'usage_count': cached['usage_count'],
                'tokens_saved': 5000,  # Estimativa dos tokens economizados
                'cost_saved': 0.0125   # Estimativa do custo economizado
            }
        
        # 2. Cache miss - chama OpenAI
        try:
            answer = await self.call_openai(question, prompt_id)
            
            # 3. Armazena para cache futuro
            point_id = await self.store_qa_pair(question, answer, user_id)
            
            return {
                'answer': answer,
                'source': 'openai',
                'point_id': point_id,
                'tokens_used': 5000,  # Estimativa
                'cost': 0.0125        # Estimativa
            }
            
        except Exception as e:
            print(f"❌ Erro no pipeline: {e}")
            return {
                'error': str(e),
                'answer': "Desculpe, ocorreu um erro ao processar sua pergunta."
            }
    
    def get_cache_stats(self):
        """
        Retorna estatísticas do cache
        
        Returns:
            Dict: Estatísticas do cache
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Busca algumas métricas básicas
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            points = scroll_result[0]
            total_usage = sum(point.payload.get('usage_count', 1) for point in points)
            avg_usage = total_usage / len(points) if points else 0
            
            return {
                'total_questions': collection_info.points_count,
                'total_usage': total_usage,
                'average_usage_per_question': round(avg_usage, 2),
                'estimated_tokens_saved': (total_usage - len(points)) * 5000,
                'estimated_cost_saved': (total_usage - len(points)) * 0.0125
            }
            
        except Exception as e:
            print(f"❌ Erro ao buscar stats: {e}")
            return {'error': str(e)}


# Exemplo de uso
async def main():
    """Exemplo de como usar o sistema"""
    
    # Inicializa o serviço (coloque sua API key)
    qa_service = QdrantOpenAIQAService(
        openai_api_key="sk-sua-chave-aqui",  # Ou use variável de ambiente
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    # Perguntas de teste (similares semanticamente)
    test_questions = [
        "Como calcular juros compostos?",
        "Qual a fórmula dos juros compostos?", 
        "Me explica o cálculo de juros compostos",
        "Juros compostos como funciona?",
        "Como fazer cálculo de rendimento composto?"
    ]
    
    print("🚀 Iniciando teste do sistema Q&A")
    print("=" * 50)
    
    # Testa cada pergunta
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- TESTE {i}/5 ---")
        
        result = await qa_service.answer_question(
            question=question,
            user_id=f"user_{i}",
            similarity_threshold=0.8  # 80% de similaridade
        )
        
        if 'error' not in result:
            print(f"📋 Fonte: {result['source']}")
            if result['source'] == 'cache':
                print(f"🎯 Similaridade: {result['similarity']:.4f}")
                print(f"💰 Tokens economizados: {result['tokens_saved']}")
            else:
                print(f"💸 Tokens gastos: {result['tokens_used']}")
            
            print(f"💬 Resposta: {result['answer'][:150]}...")
        else:
            print(f"❌ Erro: {result['error']}")
        
        # Pequena pausa entre requests
        await asyncio.sleep(1)
    
    # Mostra estatísticas finais
    print(f"\n{'='*50}")
    print("📊 ESTATÍSTICAS FINAIS")
    print("=" * 50)
    
    stats = qa_service.get_cache_stats()
    if 'error' not in stats:
        print(f"📝 Total de perguntas armazenadas: {stats['total_questions']}")
        print(f"🔄 Total de usos: {stats['total_usage']}")
        print(f"📈 Uso médio por pergunta: {stats['average_usage_per_question']}")
        print(f"💰 Tokens economizados: {stats['estimated_tokens_saved']:,}")
        print(f"💵 Custo economizado: ${stats['estimated_cost_saved']:.4f}")
    else:
        print(f"❌ Erro nas estatísticas: {stats['error']}")

if __name__ == "__main__":
    # Para testar, rode:
    # python script.py
    asyncio.run(main())