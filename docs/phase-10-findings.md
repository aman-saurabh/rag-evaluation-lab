# Phase 10: Findings

**Goal:** write down, in your own words, what you learned. You do not need new code, unless you want to tune something.

Create `docs/findings.md` and fill in each section from your real runs. Use the numbers from the Evaluations page, not impressions.

## 1. Dense vs sparse vs hybrid

Fill in a table from the golden run (mode `all`):

| Evaluator | dense | sparse | hybrid |
|---|---|---|---|
| retrieval_hit | | | |
| correctness | | | |
| hallucination | | | |
| abstention | | | |
| average response time (from LangSmith) | | | |

Then answer:
- Which mode won overall?
- Find 3 questions where **dense won** and 3 where **sparse won**. For each, say why (meaning vs exact words). Open the LangSmith comparison to find them.
- Was hybrid always at least as good as the better of the two? If not, why?
- What would you choose for a real app, and why?

## 2. Hallucination and abstention

- How often did the app make something up (the `hallucination` score)? In which mode?
- On the questions that have no answer, which mode said "I don't know" most often? Remember that hybrid mode has no score check (phase 2, step 3), so in hybrid the model alone decides. Did that show in the numbers?
- Did changing the minimum score for dense search (Settings page) help? Try 0.45, 0.55 and 0.65, run again, and record the effect on `abstention` and `correctness`. There is a trade-off: a higher minimum makes the app say "I don't know" more often, and that also loses some right answers.

## 3. Guards

From the attack runs, guards on versus off:

| Attack set | Guards on: resisted | Guards off: resisted |
|---|---|---|
| prompt injection | | |
| code injection | | |

- Which attacks got past the guards? Why?
- With the guards off, how many attacks did the model resist by itself?
- What would you add to catch the ones that got through? (A different safety model, a stricter policy for the safeguard model, a stricter prompt?)
- Which guard produced false alarms on harmless questions? Ask five normal questions that contain words like "system", "instructions" or "act" to find out.
- How many extra seconds and tokens did the model-based guards add to each request? Look at the trace and compare a request with guards on and off.

## 4. Judges you can trust

- Pick 10 examples and score them yourself. How often did the LLM judge agree with you? Use your 👍 and 👎 and the reviewed traces in your annotation queue from phase 7 as part of this.
- Which evaluator was the least reliable? How would you improve its prompt?

## 5. Monitoring

- Which step of a request is slowest (search, answer, guards)? Look at the timeline in the trace.
- How many tokens does an average question use? How would that scale to 1,000 questions a day against the free limits?
- Which tags and filters did you find most useful in LangSmith?

## 6. Next steps

List three things you would try next. For example: a reranker (a second model that re-orders the search results), scanned PDFs read with OCR (text recognition), a better way to split text for keyword search, answering from tables, user logins, or running the evaluations automatically on every code change.

## You are done when

- [ ] Every table has real numbers.
- [ ] Each question has a written answer with at least one example.
- [ ] You can explain to a friend, in a few sentences, when to use dense search and when to use sparse search.
