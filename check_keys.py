import os
import httpx
from groq import Groq
from langsmith import Client
from app.config import EMBED_URL

print("Groq:", Groq().chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Say ok"}],
    max_tokens=300,
).choices[0].message.content)

r = httpx.post(EMBED_URL, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
               json={"inputs": ["hello"]}, timeout=60)
print("HuggingFace:", r.status_code, len(r.json()[0]))

print("LangSmith:", [p.name for p in Client().list_projects(limit=3)])
