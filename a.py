# https://www.youtube.com/watch?v=KRiFMFVPL1Q
# qdrant
from turtle import distance
from typing import Collection
from qdrant_client import QdrantClient

# Criar nossa coleção
from qdrant_client.http.models import Distance, VectorParams

# lanchaing (Bosta) substituri por sentence_transformers
# from langchain.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore

# openai embedding
from langchain_openai import OpenAIEmbeddings

# gera chunks
from langchain.text_splitter import CharacterTextSplitter

# Usa vetorer para perguntas
from langchain.chains import RetrievalQA

# api openai
from langchain_openai import OpenAI

import os

client = QdrantClient("http://localhost:6333")

# client.create_collection(
#     collection_name="openai_c",
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
# )

# importe a variavel de ambiente
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = QdrantVectorStore(
    client=client, collection_name="openai_c", embedding=embeddings
)


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        # is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks


# with open("estrutura_banco.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

#     texts = get_chunks(raw_text)

#     vectorstore.add_texts(texts=texts)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4.1-nano-2025-04-14"), chain_type="stuff", retriever=vectorstore.as_retriever()
)

result = qa.invoke("Mostre as chaves estrangeiras da tabela channel")
print(result['result'])
