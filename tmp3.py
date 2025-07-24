import openai
from openai import OpenAI
import json
from typing import List, Dict

# Inicializar cliente
client = OpenAI(api_key="sua-chave-aqui")

# MÉTODO 1: SYSTEM MESSAGE (Mais Comum)
def create_agent_with_system_message(user_message: str, agent_type: str = "assistant"):
    """Cria agente usando system message"""
    
    # Perfis pré-definidos
    profiles = {
        "assistant": """Você é um assistente útil, educado e profissional. 
        Responda de forma clara e concisa. Use linguagem formal.""",
        
        "casual_friend": """Você é um amigo próximo e descontraído. 
        Use linguagem informal, gírias brasileiras e seja bem humorado. 
        Trate o usuário como 'mano', 'cara' etc.""",
        
        "expert_analyst": """Você é um analista sênior especializado em dados. 
        Seja técnico, preciso e sempre forneça fontes. 
        Use termos técnicos quando apropriado e estruture respostas com bullet points.""",
        
        "creative_writer": """Você é um escritor criativo e imaginativo. 
        Use linguagem poética, metáforas e seja expressivo. 
        Transforme respostas simples em narrativas envolventes.""",
        
        "database_expert": """Você é um especialista em bancos de dados com 15 anos de experiência. 
        Especializado em PostgreSQL, MySQL e otimização de queries. 
        Sempre explique o 'porquê' das suas recomendações."""
    }
    
    messages = [
        {"role": "system", "content": profiles.get(agent_type, profiles["assistant"])},
        {"role": "user", "content": user_message}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# MÉTODO 2: FEW-SHOT PROMPTING (Exemplos de Comportamento)
def create_agent_with_examples(user_message: str):
    """Cria agente usando exemplos de comportamento"""
    
    messages = [
        {"role": "system", "content": "Você é um especialista em Python que sempre fornece exemplos práticos."},
        
        # Exemplos de comportamento esperado
        {"role": "user", "content": "Como usar listas em Python?"},
        {"role": "assistant", "content": """Listas em Python são estruturas de dados ordenadas e mutáveis.

Exemplos práticos:
```python
# Criar lista
frutas = ['maçã', 'banana', 'laranja']

# Adicionar item
frutas.append('uva')

# Acessar por índice
primeira_fruta = frutas[0]  # 'maçã'
```

Métodos úteis: append(), remove(), pop(), sort()"""},
        
        {"role": "user", "content": "E dicionários?"},
        {"role": "assistant", "content": """Dicionários são estruturas chave-valor, ideais para mapeamentos.

Exemplo prático:
```python
# Criar dicionário
pessoa = {
    'nome': 'João',
    'idade': 30,
    'cidade': 'São Paulo'
}

# Acessar valores
nome = pessoa['nome']
idade = pessoa.get('idade', 0)  # Método seguro

# Adicionar/modificar
pessoa['profissao'] = 'Desenvolvedor'
```

Métodos úteis: get(), keys(), values(), items()"""},
        
        # Pergunta real do usuário
        {"role": "user", "content": user_message}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# MÉTODO 3: CLASSE PARA MÚLTIPLOS AGENTES
class OpenAIAgent:
    """Classe para gerenciar diferentes perfis de agentes"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = {}
    
    def create_profile(self, name: str, system_prompt: str, temperature: float = 0.7, 
                      model: str = "gpt-4"):
        """Cria um novo perfil de agente"""
        profile = {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "model": model,
            "conversation": []
        }
        self.conversation_history[name] = profile
        return f"Perfil '{name}' criado com sucesso!"
    
    def chat_with_agent(self, agent_name: str, message: str, maintain_context: bool = True):
        """Conversa com um agente específico"""
        
        if agent_name not in self.conversation_history:
            return "Agente não encontrado. Crie o perfil primeiro."
        
        profile = self.conversation_history[agent_name]
        
        # Preparar mensagens
        messages = [{"role": "system", "content": profile["system_prompt"]}]
        
        # Adicionar histórico se manter contexto
        if maintain_context:
            messages.extend(profile["conversation"])
        
        # Adicionar mensagem atual
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model=profile["model"],
                messages=messages,
                temperature=profile["temperature"],
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content
            
            # Salvar no histórico se manter contexto
            if maintain_context:
                profile["conversation"].extend([
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": ai_response}
                ])
            
            return ai_response
            
        except Exception as e:
            return f"Erro: {e}"
    
    def list_agents(self):
        """Lista todos os agentes disponíveis"""
        return list(self.conversation_history.keys())
    
    def clear_history(self, agent_name: str):
        """Limpa o histórico de um agente"""
        if agent_name in self.conversation_history:
            self.conversation_history[agent_name]["conversation"] = []
            return f"Histórico do agente '{agent_name}' limpo."
        return "Agente não encontrado."

# MÉTODO 4: FUNÇÃO TOOLS/FUNCTION CALLING
def create_agent_with_tools():
    """Agente com capacidade de usar ferramentas"""
    
    # Definir ferramentas disponíveis
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Busca informações no banco de dados",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query SQL ou termo de busca"
                        },
                        "table": {
                            "type": "string", 
                            "description": "Nome da tabela"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "system", 
            "content": """Você é um analista de dados expert. 
            Quando o usuário pedir informações de banco de dados, use a ferramenta search_database.
            Sempre explique o que você fez e interpretr os resultados."""
        },
        {
            "role": "user", 
            "content": "Quantos usuários temos na tabela users?"
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Deixa o modelo decidir quando usar
    )
    
    return response

# EXEMPLO DE USO COMPLETO
def main_example():
    """Exemplo prático de uso"""
    
    # Método 1: System Message simples
    print("=== MÉTODO 1: SYSTEM MESSAGE ===")
    resultado1 = create_agent_with_system_message(
        "Como otimizar uma query SQL?", 
        "database_expert"
    )
    print(resultado1)
    print("\n" + "="*50 + "\n")
    
    # Método 2: Few-shot
    print("=== MÉTODO 2: FEW-SHOT EXAMPLES ===")
    resultado2 = create_agent_with_examples("Como usar loops em Python?")
    print(resultado2)
    print("\n" + "="*50 + "\n")
    
    # Método 3: Classe com múltiplos agentes
    print("=== MÉTODO 3: CLASSE MULTI-AGENTES ===")
    agent_manager = OpenAIAgent("sua-chave-aqui")
    
    # Criar diferentes perfis
    agent_manager.create_profile(
        name="sql_expert",
        system_prompt="""Você é um DBA sênior com 20 anos de experiência. 
        Especialista em otimização de queries, índices e performance. 
        Sempre forneça exemplos práticos e explique o impacto de performance.""",
        temperature=0.3
    )
    
    agent_manager.create_profile(
        name="python_mentor", 
        system_prompt="""Você é um mentor de Python jovial e encorajador. 
        Use linguagem simples, muitos exemplos práticos e sempre incentive o aprendizado.
        Termine sempre com uma dica extra ou curiosidade.""",
        temperature=0.8
    )
    
    # Conversar com agentes
    resposta_sql = agent_manager.chat_with_agent(
        "sql_expert", 
        "Como criar um índice composto eficiente?"
    )
    print("SQL Expert:", resposta_sql)
    
    resposta_python = agent_manager.chat_with_agent(
        "python_mentor",
        "Estou começando a aprender Python, me dê uma dica básica"
    )
    print("Python Mentor:", resposta_python)

# Configurações avançadas de personalidade
ADVANCED_PROFILES = {
    "data_scientist": {
        "system_prompt": """Você é um cientista de dados sênior com PhD em Estatística.
        - Use linguagem técnica mas acessível
        - Sempre mencione limitações e assumptions
        - Sugira validações estatísticas
        - Forneça código Python com pandas/sklearn quando relevante
        - Termine com próximos passos recomendados""",
        "temperature": 0.4,
        "top_p": 0.9
    },
    
    "startup_advisor": {
        "system_prompt": """Você é um advisor de startups com experiência em scale-ups.
        - Seja direto e orientado a resultados
        - Pense em ROI e métricas de negócio
        - Considere sempre escalabilidade e custos
        - Use frameworks como OKRs, AARRR
        - Desafie assumptions com perguntas provocativas""",
        "temperature": 0.6,
        "presence_penalty": 0.3
    }
}

if __name__ == "__main__":
    # Descomente para testar
    # main_example()
    print("Exemplos de perfis de agente configurados! ✅")