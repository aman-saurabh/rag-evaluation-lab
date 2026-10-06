from langchain_groq import ChatGroq
from app.config import FAST_MODEL, STRONG_MODEL, PROMPT_GUARD_MODEL, SAFEGUARD_MODEL

fast_llm = ChatGroq(model=FAST_MODEL, temperature=0, max_retries=3, max_tokens=2000)
strong_llm = ChatGroq(model=STRONG_MODEL, temperature=0, max_retries=3, max_tokens=2000)

# Models for the guards (phase 6)
prompt_guard_llm = ChatGroq(model=PROMPT_GUARD_MODEL, temperature=0)
# The free tier allows only 2,000 tokens per minute for this model, and Groq counts max_tokens as part of
# every request. So keep max_tokens small and let the model think as little as possible.
safeguard_llm = ChatGroq(model=SAFEGUARD_MODEL, temperature=0, max_retries=3, max_tokens=600,
                         reasoning_effort="low")
