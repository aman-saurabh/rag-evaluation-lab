# Phase 10: Findings

**Goal:** answer, in writing, what you set out to learn. No new code unless you want to tune something.

Create `docs/findings.md` and fill in each section from your real runs. Use numbers from the Evaluations page, not feelings.

## 1. Dense vs sparse vs hybrid

Fill in a table from the golden run (mode `all`):

| Evaluator | dense | sparse | hybrid |
|---|---|---|---|
| retrieval_hit | | | |
| answer_correct | | | |
| grounded | | | |
| abstention | | | |
| average latency (from LangSmith) | | | |

Then answer:
- Which mode won overall?
- Find 3 questions where **dense won** and 3 where **sparse won**. For each, say why (meaning vs exact words). Open the LangSmith comparison to find them.
- Was hybrid always at least as good as the better of the two? If not, why?
- What would you choose for a real app, and why?

## 2. Hallucination and abstention

- How often did the app make something up (the `grounded` score)? In which mode?
- On the unanswerable questions, which mode abstained most? Did tuning the dense score threshold (Settings page) help? Try 0.45, 0.55 and 0.65, rerun, and record the effect on `abstention` and `answer_correct`. There is a trade-off: a higher threshold abstains more, which also loses some right answers.

## 3. Guards

From the attack runs, guards on versus off:

| Attack set | Guards on: resisted | Guards off: resisted |
|---|---|---|
| prompt injection | | |
| code injection | | |

- Which attacks got past the guards? Why?
- With the guards off, how many attacks did the model resist by itself?
- What would you add to catch the ones that got through? (A better phrase list, an AI-based check, a stricter prompt?)
- Which guard produced false alarms on harmless questions? Ask five normal questions that contain words like "system", "instructions" or "act" to find out.

## 4. Judges you can trust

- Pick 10 examples and score them yourself. How often did the LLM judge agree with you? Use your 👍 and 👎 from phase 7 as part of this.
- Which evaluator was the least reliable? How would you improve its prompt?

## 5. Monitoring

- Which step of a request is slowest (search, answer, guards)? Look at the trace waterfall.
- How many tokens does an average question use? How would that scale to 1,000 questions a day against the free limits?
- Which tags and filters did you find most useful in LangSmith?

## 6. Next steps

List three things you would try next. For example: a reranker, scanned PDFs with OCR, a better sparse tokenizer, answering from tables, user logins, or running the evals automatically on every code change.

## You are done when

- [ ] Every table has real numbers.
- [ ] Each question has a written answer with at least one example.
- [ ] You can explain to a friend, in a few sentences, when to use dense search and when to use sparse search.
