# Phase 9: Run evaluations from the web page

**Goal:** on the Evaluations page, pick a dataset, a search mode and the evaluators, press Run, watch a progress bar, and see the scores.

## The problem to solve

A run takes minutes: about 24 questions, each with an answer and several judge calls, all under Groq's token limit. A web request cannot wait that long. So the backend starts the run **in the background** and answers straight away with a `job_id` (a ticket number). The page then asks "how far along is it?" every few seconds.

## Step 1. The runner (`app/eval_runner.py`)

`run_evaluation(job, dataset, mode, evaluators)` does this:

1. Load the YAML file for `dataset` and upload it to LangSmith as a dataset (`client.create_dataset(...)`, `client.create_examples(...)`). Use a stable name such as `rel-golden`. If it already exists, update it (delete the old examples and add the current ones), so edits made on the Datasets page take effect.
2. Call `evaluate(target, data=dataset_name, evaluators=[...], experiment_prefix=f"{dataset}-{mode}", max_concurrency=1, metadata={"mode": mode})`.
   - `max_concurrency=1` runs one example at a time, so you do not hit the Groq limit.
3. After each example, update `job["done"]`, so the page can show progress. In `langsmith` 0.14.4, `evaluate()` has no way to report progress, so count it yourself: add 1 to `job["done"]` at the end of your `target` function (and at the end of each evaluator, if you want finer progress).
4. When it is finished, read the average score of each evaluator from the experiment and store it in `job["scores"]`, together with the link to the experiment.

**Slow the calls down with a LangChain component, not a hand-written sleep.** LangChain has `InMemoryRateLimiter` (`from langchain_core.rate_limiters import InMemoryRateLimiter`). It limits how many requests per second a model may make. Create one and give it to the models in `llm.py`: `ChatGroq(..., rate_limiter=limiter)`. It counts requests, not tokens. So choose a small number, for example `requests_per_second=0.2`, and check the result against Groq's limit of 8,000 tokens per minute. `max_retries` on the models still handles the "too many requests" error (429). With `max_concurrency=1`, this keeps a run under Groq's limits.

This is the second change to `llm.py` (phase 6 added the guard models). A run also calls the guard models for every example, unless you switched the guards off in Settings, and the judges call the strong model. So a run uses a lot of tokens. Check Groq's daily limits before a full run.

## Step 2. The job list

A job is a small dictionary kept in memory:

```python
JOBS = {}   # job_id -> {"status": "running", "done": 3, "total": 24, "scores": {}, "url": None, "error": None, ...}
```

The status is `queued`, `running`, `done` or `failed`. Only one run at a time is allowed. If you start a second one while the first is running, you get a clear message "a run is already in progress", because the Groq limit is shared.

If the server restarts, the jobs in memory are lost. The experiments stay in LangSmith, which already has a page for them. So we do not build our own history table. The page links to the dataset's experiments in LangSmith.

## Step 3. The URLs (`api/routes_evals.py`)

| URL | What it does |
|---|---|
| `POST /evals/run` | Takes `dataset` (golden, attacks_prompt or attacks_code), `mode` (dense, sparse, hybrid or all) and `evaluators` (a list of names). Check every value against the allowed lists. Start the run with FastAPI's `BackgroundTasks` and return `{"job_id": "..."}`. |
| `GET /evals/{job_id}` | Returns the status, `done` and `total`, and (when finished) the scores for each mode and evaluator, and the link to the LangSmith experiment. |

`mode = "all"` runs the three modes one after another as three experiments in the same job. This is the dense vs sparse vs hybrid comparison.

Note: `BackgroundTasks` runs the work after the reply is sent, inside the same server. That is enough here. Write the work as a plain function (not `async`), so FastAPI runs it in a separate thread and the server stays responsive.

## Step 4. The Evaluations page (`ui/pages/5_Evaluations.py`)

**Start a run:**
- A dataset selector, a mode selector (including "all") and a multi-select of evaluators (already filled with the sensible set for the chosen dataset from phase 8).
- A warning that shows the estimated number of LLM calls and says "this takes several minutes".
- A Run button: call `POST /evals/run` and remember the `job_id` in `st.session_state`.

**Watch it:**
- While the job is `running`, show a progress bar (`st.progress(done/total)`) and refresh every 3 seconds (`time.sleep(3)` and then `st.rerun()`).
- If it `failed`, show the error text.

**Results:**
- A table: rows are the evaluators, columns are dense, sparse and hybrid. Each cell is an average score from 0 to 1.
- A link for each experiment: "Open in LangSmith" for the details of each question.
- A link to the dataset's experiments in LangSmith. History and comparison live there (select two or more experiments and use **Compare**), so we do not rebuild them.

Do not rebuild the details of each question. LangSmith's experiment page already shows every example, the reason each evaluator gave, and the trace. Just link to it.

## Step 5. Run the first full comparison

1. Dataset `golden`, mode `all`, the default evaluators. Wait.
2. Dataset `attacks_prompt`, mode `hybrid`, with all guards **on**. Note the scores.
3. Switch the guards off on the Settings page and run `attacks_prompt` again. Note the difference.
4. Do the same for `attacks_code`.
5. In LangSmith, open the three golden experiments and use **Compare** to see them side by side.

## You are done when

- [ ] You can start a run from the page and a progress bar moves.
- [ ] A finished run shows scores for each evaluator and mode, with working LangSmith links.
- [ ] You ran the attack sets with guards on and off and saw the difference.
- [ ] A second run started while one is running is refused with a clear message.
- [ ] After restarting the backend, your earlier experiments are still in LangSmith.
