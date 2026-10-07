from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from app.config import FAST_MODEL, STRONG_MODEL, PROMPT_GUARD_MODEL, SAFEGUARD_MODEL

# Phase 9: an evaluation run asks many questions in a row. A rate limiter makes each model wait between its
# requests (0.5 per second = at most one request every 2 seconds), so a run stays under Groq's limits.
# It counts requests, not tokens, so the number is a first guess: lower it if you still see "429 too many requests".
fast_limiter = InMemoryRateLimiter(requests_per_second=0.5)
strong_limiter = InMemoryRateLimiter(requests_per_second=0.5)
safeguard_limiter = InMemoryRateLimiter(requests_per_second=0.5)

fast_llm = ChatGroq(model=FAST_MODEL, temperature=0, max_retries=3, max_tokens=2000, rate_limiter=fast_limiter)
# strong_llm is the judge (the hallucination guard and the evaluators). The model thinks before it answers,
# and that thinking counts toward max_tokens. With 2000 the thinking could use everything, and the reply
# was empty ("Failed to validate JSON"). So: more room, and less thinking.
strong_llm = ChatGroq(model=STRONG_MODEL, temperature=0, max_retries=3, max_tokens=4000, reasoning_effort="low",
                    rate_limiter=strong_limiter)

# Models for the guards (phase 6)
prompt_guard_llm = ChatGroq(model=PROMPT_GUARD_MODEL, temperature=0)
# The free tier allows only 2,000 tokens per minute for this model, and Groq counts max_tokens as part of
# every request. So keep max_tokens small and let the model think as little as possible.
safeguard_llm = ChatGroq(model=SAFEGUARD_MODEL, temperature=0, max_retries=3, max_tokens=600,
                         reasoning_effort="low", rate_limiter=safeguard_limiter)
