from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index + chunks
index = faiss.read_index("faiss_index.index")
with open("faiss_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

HF_TOKEN = os.environ["HF_API_TOKEN"]
HF_MODEL = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
headers = {"Authorization": f"Bearer {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}"}

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    question = query.question
    embedding = model.encode([question])
    D, I = index.search(np.array(embedding), k=3)
    context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""You are a helpful assistant for an e-commerce site.
Use only the context below to answer the question.

[Context]
{context}

Question: {question}
Answer:"""

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json={"inputs": prompt}
    )
    return {"answer": response.json()[0]["generated_text"].replace(prompt, "").strip()}
