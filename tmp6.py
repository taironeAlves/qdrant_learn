import timeit
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import statistics
from openai import OpenAI
import time
from langchain_openai import OpenAIEmbeddings

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai():
    return client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": "Hello!"}]
    )

# Mede 1 execução
latency = timeit.timeit(call_openai, number=1)
print(f"Latência: {latency:.3f} segundos")

class OpenAILatencyMeter:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.latencies = []
    
    def measure_completion(self, model, messages, runs=5):
        for _ in range(runs):
            start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            end = time.perf_counter()
            self.latencies.append(end - start)
        
        return {
            'avg_latency': statistics.mean(self.latencies),
            'min_latency': min(self.latencies),
            'max_latency': max(self.latencies),
            'median_latency': statistics.median(self.latencies)
        }

# Uso
meter = OpenAILatencyMeter()
stats = meter.measure_completion(
    model=os.getenv("MODEL"),
    messages=[{"role": "user", "content": "Explain AI"}],
    runs=10
)
print(stats)

embeddings = OpenAIEmbeddings()

def measure_embeddings(texts):
    start = time.perf_counter()
    vectors = embeddings.embed_documents(texts, model="text-embedding-3-small")
    end = time.perf_counter()
    
    return {
        'total_time': end - start,
        'time_per_text': (end - start) / len(texts),
        'texts_processed': len(texts)
    }

# Teste
result = measure_embeddings(["Hello world", "AI is amazing"])
print(result)