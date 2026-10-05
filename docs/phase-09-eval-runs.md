# Phase 9: Run evaluations from the web page

**Goal:** on the Evaluations page, pick a dataset, a search mode and evaluators, press Run, watch a progress bar, and see the scores.

## The problem to solve

A run takes minutes (24 questions, each with an answer and several judge calls, under Groq's token limit). A web request cannot wait that long. So the backend starts the run **in the background**, answers immediately with a `job_id`, and the page asks "how far along?" every few seconds.

## Step 1. The runner (`app/eval_runner.py`)

`run_evaluation(job, dataset, mode, evaluators)` does:

1. Load the YAML for `dataset` and upload it to LangSmith as a dataset (`client.create_dataset(...)`, `client.create_examples(...)`). Use a stable name such as `rel-golden`. If it exists, update it (delete old examples and add the current ones) so the page's edits take effect.
2. Call `evaluate(target, data=dataset_name, evaluators=[...], experiment_prefix=f"{dataset}-{mode}", max_concurrency=1, metadata={"mode": mode})`.
   - `max_concurrency=1` runs one example at a time, so you do not hit the Groq limit.
3. After each example, update `job["done"]` so the page can show progress. `evaluate()` in `langsmith` 0.14.4 has no progress callback, so count progress yourself: increase `job["done"]` at the end of your `target` function (and of each evaluator, if you want finer progress).
4. When finished, read the experiment's average score per evaluator and store it in `job["scores"]` with the experiment URL.

**Pace the calls.** Before each judge call, check how many tokens you used in the last minute and sleep if you are near 6,000. A simple version: sleep 3 seconds between LLM calls. Retry on a 429 (`max_retries` on the `ChatGroq` models in `llm.py` already does this).

## Step 2. The job store

A job is a small dictionary kept in memory:

```python
JOBS = {}   # job_id -> {"status": "running", "done": 3, "total": 24, "scores": {}, "url": None, "error": None, ...}
```

Statuses: `queued`, `running`, `done`, `failed`. Only one run at a time (a second request while one is running gets a clear "a run is already in progress" error), because the Groq limit would be shared.

If the server restarts, in-memory jobs are lost. The experiments remain in LangSmith. `GET /evals` lists past experiments from LangSmith (`client.list_projects(reference_dataset_id=...)`), so history is never lost.

## Step 3. The endpoints (`api/routes_evals.py`)

| Endpoint | Does |
|---|---|
| `POST /evals/run` | Body: `dataset` (golden, attacks_prompt or attacks_code), `mode` (dense, sparse, hybrid or all), `evaluators` (a list of names). Validate every value against the allowed lists. Start the run with FastAPI's `BackgroundTasks` and return `{"job_id": "..."}`. |
| `GET /evals/{job_id}` | Returns status, `done` and `total`, scores per mode and evaluator once finished, and the LangSmith experiment link. |
| `GET /evals` | History: past experiments with name, date, mode and link. |

`mode = "all"` runs the three modes one after another as three experiments in the same job. This is the dense vs sparse vs hybrid comparison.

Note: `BackgroundTasks` runs after the response is sent but inside the same server process. That is enough here. Use a plain synchronous function so FastAPI runs it in a worker thread and the server stays responsive.

## Step 4. The Evaluations page (`ui/pages/5_Evaluations.py`)

**Start a run:**
- Dataset selector, mode selector (including "all"), a multiselect of evaluators (pre-filled with the sensible set for the chosen dataset from phase 8).
- Show a warning with the estimated number of LLM calls and "this takes several minutes".
- Run button: call `POST /evals/run`, remember `job_id` in `st.session_state`.

**Watch it:**
- While the job is `running`, show `st.progress(done/total)` and refresh every 3 seconds (`time.sleep(3)` then `st.rerun()`).
- On `failed`, show the error text.

**Results:**
- A table: rows = evaluators, columns = dense, sparse, hybrid. Cells are average scores (0 to 1).
- A link per experiment: "Open in LangSmith" for per-question detail.
- Below, a **history** table from `GET /evals`.

For per-question detail, do not rebuild it. LangSmith's experiment page shows every example, every evaluator's reason, and the trace. Link to it.

## Step 5. Run the first full comparison

1. Dataset `golden`, mode `all`, the default evaluators. Wait.
2. Dataset `attacks_prompt`, mode `hybrid`, with all guards **on**. Note the scores.
3. Switch the guards off on the Settings page and run `attacks_prompt` again. Note the difference.
4. Repeat for `attacks_code`.
5. In LangSmith, open the three golden experiments and use **Compare** to see them side by side.

## You are done when

- [ ] You can start a run from the page and a progress bar moves.
- [ ] A finished run shows scores per evaluator and mode, with working LangSmith links.
- [ ] You ran the attack sets with guards on and off and saw the difference.
- [ ] A second run started while one is running is refused with a clear message.
- [ ] Restarting the backend does not lose the history list.
