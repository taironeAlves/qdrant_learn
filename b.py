# https://www.youtube.com/watch?v=KRiFMFVPL1Q
# qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

import os

client = QdrantClient("http://localhost:6333")

# client.create_collection(
#     collection_name="b_openai",
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
# )

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = QdrantVectorStore(
    client=client, collection_name="b_openai", embedding=embeddings
)


# def get_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="Tabela: ",
#         chunk_size=1000,
#         chunk_overlap=0,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# with open("estrutura_banco.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

#     texts = get_chunks(raw_text)

#     vectorstore.add_texts(texts=texts)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4.1-nano-2025-04-14", max_tokens=150, temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# result = qa.invoke("Mostre as chaves estrangeiras da tabela channel")
result = qa.invoke("Mostre as colunas da table chat_list")
# result = qa.invoke("a table contact tem alguma relação com a tabela chat_list?")
#result = qa.invoke("a table contact tem alguma relação com a tabela chat_list?")
print(result["result"])
