import time
import httpx
import streamlit as st
import client

MODES = ["dense", "sparse", "hybrid", "all"]

st.title("Evaluations")
st.write(
    "Pick a dataset, a search mode and the evaluators, then press Run. The backend runs every question through the app "
    "and scores the answers. \"all\" runs dense, sparse and hybrid one after another, so you can compare them."
)

try:
    options = client.eval_options()
    datasets = client.list_datasets()
except httpx.ConnectError:
    st.error(client.BACKEND_DOWN_MESSAGE)
    st.stop()

# ---------- Start a run ----------
dataset_names = []
questions_in_dataset = {}
for dataset_info in datasets:
    dataset_names.append(dataset_info["name"])
    questions_in_dataset[dataset_info["name"]] = dataset_info["examples"]

dataset = st.selectbox("Dataset", dataset_names)
mode = st.selectbox("Search mode", MODES, index=3)

# The key changes with the dataset, so the list goes back to that dataset's default evaluators when you change it.
chosen_evaluators = st.multiselect(
    "Evaluators",
    options["evaluators"],
    default=options["defaults"][dataset],
    key=f"evaluators_{dataset}",
)

if mode == "all":
    number_of_modes = 3
else:
    number_of_modes = 1
number_of_questions = questions_in_dataset[dataset] * number_of_modes
st.warning(
    f"This runs {number_of_questions} questions. Each question makes several LLM calls (the answer, the guards and "
    "one judge for each judge evaluator), and Groq's free limits are slow. **This takes several minutes.**"
)

if st.button("Run"):
    try:
        started = client.start_eval(dataset, mode, chosen_evaluators)
        st.session_state.eval_job_id = started["job_id"]
        st.rerun()
    except httpx.HTTPStatusError as error:
        st.error(f"The backend refused: {error.response.text}")

# ---------- Watch the run ----------
job_id = st.session_state.get("eval_job_id")
if job_id is None:
    st.stop()

st.divider()
try:
    job = client.eval_status(job_id)
except httpx.HTTPStatusError:
    # The backend was restarted, so it forgot the job. The experiments are still in LangSmith.
    st.info("The backend does not know this run any more (it was restarted). Your experiments are still in LangSmith.")
    st.session_state.eval_job_id = None
    st.stop()

st.subheader(f"Run {job['job_id']}: {job['status']}")

if job["total"] > 0:
    share_done = job["done"] / job["total"]
else:
    share_done = 0.0
if share_done > 1.0:
    share_done = 1.0
st.progress(share_done, text=f"{job['done']} of {job['total']} questions answered")

if job["status"] == "failed":
    st.error(job["error"])

# ---------- Results ----------
if len(job["scores"]) > 0:
    st.write("Average score of each evaluator (1 is good, except `attack_detected`, which counts the attacks).")
    st.table(job["scores"])  # columns: the modes. rows: the evaluators.

    for one_mode, url in job["urls"].items():
        if url is not None:
            st.markdown(f"[Open the {one_mode} experiment in LangSmith]({url})")
    if job["dataset_url"] is not None:
        st.markdown(f"[All experiments of this dataset in LangSmith (select two or more and press **Compare**)]({job['dataset_url']})")

# While the job runs, ask again in 3 seconds.
if job["status"] == "queued" or job["status"] == "running":
    time.sleep(3)
    st.rerun()
