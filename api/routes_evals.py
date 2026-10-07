from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.schemas import EvalRunRequest
from app.eval_runner import JOBS, a_run_is_active, create_job, run_job
from app.evaluators import ALL_EVALUATORS, DEFAULT_EVALUATORS

router = APIRouter()


@router.get("/evals/options")
def read_options():
    """What the page can choose from: all evaluators, and the default ones for each dataset.
    (This must stay above /evals/{job_id}, or "options" would be taken as a job id.)"""
    return {"evaluators": list(ALL_EVALUATORS), "defaults": DEFAULT_EVALUATORS}


@router.post("/evals/run")
def start_run(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """Starts a run in the background and answers at once with a job_id (a ticket number)."""
    if len(request.evaluators) == 0:
        raise HTTPException(status_code=400, detail="Choose at least one evaluator")

    for evaluator_name in request.evaluators:
        if evaluator_name not in ALL_EVALUATORS:
            raise HTTPException(status_code=400, detail=f"Unknown evaluator {evaluator_name!r}. Use some of {list(ALL_EVALUATORS)}")

    if a_run_is_active():
        raise HTTPException(status_code=409, detail="A run is already in progress. Wait until it is finished.")

    job = create_job(request.dataset, request.mode)
    # run_job is a plain function (not async), so FastAPI runs it in a separate thread and the server stays free.
    background_tasks.add_task(run_job, job, request.dataset, request.mode, request.evaluators)
    return {"job_id": job["job_id"]}


@router.get("/evals/{job_id}")
def read_job(job_id: str):
    """The status, the progress and (when finished) the scores and the LangSmith links."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="No such job. The backend may have been restarted. Your experiments are still in LangSmith.")
    return JOBS[job_id]
