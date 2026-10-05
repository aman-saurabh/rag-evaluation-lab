from langchain_groq import ChatGroq
from app.config import FAST_MODEL, STRONG_MODEL

fast_llm = ChatGroq(model=FAST_MODEL, temperature=0, max_retries=3, max_tokens=2000)
strong_llm = ChatGroq(model=STRONG_MODEL, temperature=0, max_retries=3, max_tokens=2000)
