import uuid
import yaml
from langsmith import evaluate
from app.config import DATASET_DIR
from app.evaluators import ALL_EVALUATORS, DEFAULT_EVALUATORS
from app.graph import ask
from app.tracing import langsmith_client

# The only datasets that exist. Each one is the file data/datasets/<name>.yaml.
DATASETS = ["golden", "attacks_prompt", "attacks_code"]


def load_items(dataset: str) -> list[dict]:
    """Reads a dataset file (a list of questions or attacks)."""
    if dataset not in DATASETS:  # never build a file path from a name we do not know
        raise ValueError(f"Unknown dataset {dataset!r}. Use one of {DATASETS}")

    with open(DATASET_DIR / (dataset + ".yaml"), encoding="utf-8") as dataset_file:
        return yaml.safe_load(dataset_file)


def save_items(dataset: str, items: list[dict]) -> None:
    """Writes the list of questions or attacks back to the dataset file. (Comments in the file are not kept.)"""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Use one of {DATASETS}")

    with open(DATASET_DIR / (dataset + ".yaml"), "w", encoding="utf-8") as dataset_file:
        yaml.safe_dump(items, dataset_file, sort_keys=False, allow_unicode=True, width=100)


def upload_dataset(dataset: str) -> str:
    """Uploads the file to LangSmith as the dataset "rag-lab-<name>", and replaces its old examples.
    Returns the name of the dataset in LangSmith."""
    dataset_name = "rag-lab-" + dataset

    examples = []
    for item in load_items(dataset):
        if dataset == "golden":
            # inputs = what the app gets. outputs = the right answer (the evaluators compare with it).
            example = {
                "inputs": {"question": item["question"]},
                "outputs": {"answer": item["answer"], "sources": item.get("sources", [])},
                "metadata": {"id": item["id"], "type": item["type"]},
            }
        else:
            # An attack has no right answer. We use the attack as the question.
            example = {
                "inputs": {"question": item["attack"]},
                "metadata": {"id": item["id"]},
            }
        examples.append(example)

    if langsmith_client.has_dataset(dataset_name=dataset_name):
        # The dataset exists: delete its old examples, so the file is the only truth.
        old_example_ids = []
        for old_example in langsmith_client.list_examples(dataset_name=dataset_name):
            old_example_ids.append(old_example.id)
        if len(old_example_ids) > 0:
            langsmith_client.delete_examples(old_example_ids)
    else:
        langsmith_client.create_dataset(dataset_name, description=f"Test data from data/datasets/{dataset}.yaml")

    langsmith_client.create_examples(dataset_name=dataset_name, examples=examples)
    return dataset_name


def make_target(mode: str, job: dict | None = None):
    """Makes the function that evaluate() calls for every test question. The mode is fixed inside it,
    so one dataset can be used for dense, sparse and hybrid.
    job: if given, we add 1 to job["done"] after every question (evaluate() has no progress report of its own)."""

    def target(inputs: dict) -> dict:
        # No session id: every test question is a fresh chat. The tag lets you filter these traces in LangSmith.
        result = ask(inputs["question"], mode, tags=["source:eval"])
        if job is not None:
            job["done"] = job["done"] + 1
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "abstained": result["abstained"],
            "blocked": result["blocked"],
            "guard_results": result["guard_results"],
            "retrieved_docs": result["retrieved_docs"],
        }

    return target


def run_evaluation(dataset: str, mode: str, evaluator_names: list[str] | None = None, limit: int | None = None,
                   job: dict | None = None):
    """Runs the app on every example of the dataset and scores the results. The result is an "experiment" in LangSmith.
    limit: only run the first N examples (to try things out). job: the job to report progress to (phase 9)."""
    if evaluator_names is None:
        evaluator_names = DEFAULT_EVALUATORS[dataset]

    evaluators = []
    for evaluator_name in evaluator_names:
        if evaluator_name not in ALL_EVALUATORS:
            raise ValueError(f"Unknown evaluator {evaluator_name!r}. Use some of {list(ALL_EVALUATORS)}")
        evaluators.append(ALL_EVALUATORS[evaluator_name])

    dataset_name = upload_dataset(dataset)

    if limit is None:
        data = dataset_name
    else:
        data = langsmith_client.list_examples(dataset_name=dataset_name, limit=limit)

    return evaluate(
        make_target(mode, job),
        data=data,
        evaluators=evaluators,
        experiment_prefix=f"{dataset}-{mode}",
        max_concurrency=1,  # one example at a time, so we do not hit the Groq limits
        metadata={"mode": mode},
        client=langsmith_client,
    )


# ---------- Background jobs (phase 9) ----------

MODES = ["dense", "sparse", "hybrid"]

# job_id -> {"status", "done", "total", "scores", "urls", "dataset_url", "error"}. Kept in memory only.
# The experiments themselves stay in LangSmith when the server restarts.
JOBS = {}


def a_run_is_active() -> bool:
    """True if a job is waiting or running. Only one run at a time, because Groq's limits are shared."""
    for job in JOBS.values():
        if job["status"] == "queued" or job["status"] == "running":
            return True
    return False


def create_job(dataset: str, mode: str) -> dict:
    """Makes a new job in the list. mode "all" means the three modes one after another."""
    if mode == "all":
        modes = MODES
    else:
        modes = [mode]
    total = len(load_items(dataset)) * len(modes)

    job = {
        "job_id": uuid.uuid4().hex[:8],
        "status": "queued",
        "done": 0,
        "total": total,
        "scores": {},         # {"dense": {"retrieval_hit": 0.8, ...}, "sparse": {...}}
        "urls": {},           # {"dense": link to the experiment in LangSmith, ...}
        "dataset_url": None,  # link to the dataset (all its experiments) in LangSmith
        "error": None,
    }
    JOBS[job["job_id"]] = job
    return job


def average_scores(results) -> dict:
    """The average score of each evaluator over all the questions of one experiment.
    An empty score (an evaluator that did not apply to a question) is left out."""
    totals = {}
    counts = {}
    for row in results:
        for evaluation in row["evaluation_results"]["results"]:
            if evaluation.score is None:
                continue
            key = evaluation.key
            totals[key] = totals.get(key, 0) + evaluation.score
            counts[key] = counts.get(key, 0) + 1

    averages = {}
    for key in totals:
        averages[key] = round(totals[key] / counts[key], 2)
    return averages


def run_job(job: dict, dataset: str, mode: str, evaluator_names: list[str]) -> None:
    """The work of one job. FastAPI runs it in the background, after the reply was sent."""
    if mode == "all":
        modes = MODES
    else:
        modes = [mode]

    job["status"] = "running"
    try:
        for one_mode in modes:
            results = run_evaluation(dataset, one_mode, evaluator_names, job=job)
            job["scores"][one_mode] = average_scores(results)
            job["urls"][one_mode] = results.url
            if results.url is not None:
                job["dataset_url"] = results.url.split("/compare")[0]
        job["status"] = "done"
    except Exception as error:
        job["status"] = "failed"
        job["error"] = f"{type(error).__name__}: {error}"
