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

embed_response = httpx.post(EMBED_URL, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
                            json={"inputs": ["hello"]}, timeout=60)
print("HuggingFace:", embed_response.status_code, len(embed_response.json()[0]))

print("LangSmith:", [project.name for project in Client().list_projects(limit=3)])
