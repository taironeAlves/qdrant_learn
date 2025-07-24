from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import hashlib
import uuid
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv


load_dotenv()


class ArmazenaPerguntas:
    def __init__(self):
        self.client_openai = OpenAI()
        self.client = QdrantClient("http://localhost:6333")
        self.collection_name = "perguntas_respostas"
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        self.setup_collection()

    def setup_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' já existe")
        except Exception:
            print(f"🔧 Criando collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension, distance=Distance.COSINE
                ),
            )
            print(f"✅ Collection '{self.collection_name}' criada com sucesso!")

    async def perguntas_respostas(
        self, question, user_id, prompt_id=None, similarity_threshold=0.85
    ):
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

        # 1. Verifica no banco se a pergunta ja foi feita
        cached = await self.procura_no_historico(question, similarity_threshold)
        if cached:
            return {
                "answer": cached["answer"],
                "source": "cache",
                "similarity": cached["similarity"],
                "original_question": cached["original_question"],
                "usage_count": cached["usage_count"],
                "tokens_saved": 5000,  # Estimativa dos tokens economizados
                "cost_saved": 0.0125,  # Estimativa do custo economizado
            }

        # TODO: implementar o OpenAI() se não tiver resposta no cache chama a openai pra gerar resposta e ai a resposta é gravada.
        try:
            answer = await self.call_openai(question, prompt_id)

            # 3. Armazena para cache futuro
            point_id = await self.store_qa_pair(question, answer, user_id)

            return {
                "answer": answer,
                "source": "openai",
                "point_id": point_id,
                "tokens_used": 5000,  # Estimativa
                "cost": 0.0125,  # Estimativa
            }

        except Exception as e:
            print(f"❌ Erro no pipeline: {e}")
            return {
                "error": str(e),
                "answer": "Desculpe, ocorreu um erro ao processar sua pergunta.",
            }

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
            vector = await self.encode_pergunta(question)

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
                    "embedding_model": self.embedding_model,
                },
            )

            # CORREÇÃO: Usar self.client em vez de self.qdrant_client
            self.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            print(f"💾 Q&A armazenado com ID: {point_id}")
            return point_id

        except Exception as e:
            print(f"❌ Erro ao armazenar: {e}")
            raise

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
            # Se você tem um prompt personalizado, use o ID fornecido
            if prompt_id:
                response = self.client_openai.responses.create(
                    prompt={
                        "id": prompt_id,
                    },
                    input=question,
                )
                answer = response.output_text
            else:
                # Usar o prompt padrão que você definiu no código original
                response = self.client_openai.responses.create(
                    prompt={
                        "id": "pmpt_687b097121dc8197a7c2de012e58227a00fb5a28ba6fcda4",
                    },
                    input=question,
                )
                answer = response.output_text

            print(f"🤖 Resposta OpenAI gerada: {len(answer)} caracteres")
            return answer

        except Exception as e:
            print(f"❌ Erro na chamada OpenAI: {e}")
            raise

    async def procura_no_historico(self, question, similarity_threshold=0.85, limit=3):
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
            query_vector = await self.encode_pergunta(question)

            # Usar query_points (search está depreciado)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=similarity_threshold,
                with_payload=True,
            )

            # Na v1.14.3, os resultados estão em response.points
            results = (
                response.points if response and hasattr(response, "points") else []
            )

            # Verificar se temos resultados e se o score é suficiente
            if results and len(results) > 0:
                best_match = results[0]

                # Verificar se o score atende ao threshold (dupla verificação)
                if best_match.score >= similarity_threshold:
                    # Incrementa contador de uso
                    await self.increment_usage(best_match.id)

                    print(f"✅ Cache HIT! Score: {best_match.score:.4f}")
                    print(
                        f"📝 Pergunta original: {best_match.payload['question'][:100]}..."
                    )

                    return {
                        "answer": best_match.payload["answer"],
                        "similarity": best_match.score,
                        "cached": True,
                        "original_question": best_match.payload["question"],
                        "usage_count": best_match.payload.get("usage_count", 1),
                        "point_id": best_match.id,
                    }

            # Se chegou aqui, não teve match suficiente
            best_score = results[0].score if results else 0.0
            print(f"❌ Cache MISS - Melhor score: {best_score:.4f}")
            return None

        except Exception as e:
            print(f"❌ Erro na busca cache: {e}")
            print(f"Tipo do erro: {type(e)}")
            import traceback

            traceback.print_exc()
            return None

    async def increment_usage(self, point_id):
        """
        Incrementa contador de uso de uma resposta

        Args:
            point_id: ID do ponto no Qdrant
        """
        try:
            # Pega dados atuais
            points = self.client.retrieve(
                collection_name=self.collection_name, ids=[point_id], with_payload=True
            )
            
            if not points:
                print(f"❌ Ponto {point_id} não encontrado")
                return
                
            point = points[0]

            # Incrementa usage_count
            current_count = point.payload.get("usage_count", 1)
            new_payload = point.payload.copy()
            new_payload["usage_count"] = current_count + 1
            new_payload["updated_at"] = datetime.now().isoformat()

            # Atualiza no Qdrant
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=new_payload,
                points=[point_id],
            )

            print(f"📊 Usage count atualizado: {current_count} → {current_count + 1}")

        except Exception as e:
            print(f"❌ Erro ao incrementar usage: {e}")

    async def encode_pergunta(self, question):
        """
        Converte pergunta em embedding usando OpenAI

        Args:
            question: Pergunta em texto

        Returns:
            List: Vector embedding da pergunta
        """
        try:
            response = self.client_openai.embeddings.create(
                model="text-embedding-3-small", input=question.strip()
            )

            embedding = response.data[0].embedding
            print(f"🔤 Embedding gerado: {len(embedding)} dimensões")
            return embedding

        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
            raise

    # CORREÇÃO: Renomear para encode_pergunta para consistência
    async def encode_question(self, question):
        """Alias para encode_pergunta para compatibilidade"""
        return await self.encode_pergunta(question)

    def get_cache_stats(self):
        """
        Retorna estatísticas do cache

        Returns:
            Dict: Estatísticas do cache
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)

            # Busca algumas métricas básicas
            scroll_result = self.client.scroll(
                collection_name=self.collection_name, limit=1000, with_payload=True
            )

            points = scroll_result[0]  # scroll retorna (points, next_page_offset)
            total_usage = sum(point.payload.get("usage_count", 1) for point in points)
            avg_usage = total_usage / len(points) if points else 0

            return {
                "total_questions": collection_info.points_count,
                "total_usage": total_usage,
                "average_usage_per_question": round(avg_usage, 2),
                "estimated_tokens_saved": (total_usage - len(points)) * 5000,
                "estimated_cost_saved": (total_usage - len(points)) * 0.0125,
            }

        except Exception as e:
            print(f"❌ Erro ao buscar stats: {e}")
            return {"error": str(e)}


# Essa função será usada para consumir a classe ArmazenaPerguntas
async def main():
    """Iniciando o sistema Q&A"""

    qa_service = ArmazenaPerguntas()

    interacao = [
        "Mostre as chaves estrangeiras da tabela channel",
        "Mostre as colunas da table chat_list",
        "A table contact tem alguma relação com a tabela chat_list?",
        "A table contact tem alguma relação com a tabela channel?",
    ]

    print("🚀 Iniciando teste do sistema Q&A")
    print("=" * 50)

    # Percorre uma pergunta por vez
    for i, question in enumerate(interacao, 1):
        print(f"\n--- TESTE {i}/4 ---")

        result = await qa_service.perguntas_respostas(
            question=question, user_id=f"user_{i}", similarity_threshold=0.8
        )

        # CORREÇÃO: Verificar se result não é None antes de acessar
        if result is not None and "error" not in result:
            print(f"📋 Fonte: {result['source']}")
            if result["source"] == "cache":
                print(f"🎯 Similaridade: {result['similarity']:.4f}")
                print(f"💰 Tokens economizados: {result['tokens_saved']}")
            elif result["source"] == "openai_needed":
                print(f"⚠️ {result['message']}")
            else:
                print(f"💸 Tokens gastos: {result.get('tokens_used', 'N/A')}")

            print(f"💬 Resposta: {result.get('answer', 'Sem resposta')[:150]}...")
        else:
            print(
                f"❌ Erro: {result.get('error', 'Resultado None') if result else 'Resultado None'}"
            )

        # Pequena pausa entre requests
        await asyncio.sleep(1)

    print(f"\n{'='*50}")
    print("📊 ESTATÍSTICAS FINAIS")
    print("=" * 50)

    stats = qa_service.get_cache_stats()
    if stats and "error" not in stats:
        print(f"📝 Total de perguntas armazenadas: {stats['total_questions']}")
        print(f"🔄 Total de usos: {stats['total_usage']}")
        print(f"📈 Uso médio por pergunta: {stats['average_usage_per_question']}")
        print(f"💰 Tokens economizados: {stats['estimated_tokens_saved']:,}")
        print(f"💵 Custo economizado: ${stats['estimated_cost_saved']:.4f}")
    else:
        print(
            f"❌ Erro nas estatísticas: {stats.get('error', 'Stats None') if stats else 'Stats None'}"
        )


if __name__ == "__main__":
    asyncio.run(main())