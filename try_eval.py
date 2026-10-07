from langchain_core.tracers.langchain import wait_for_all_tracers
from app.eval_runner import run_evaluation
from app.retrieve import store

# Runs the first 3 golden questions in hybrid mode and prints every evaluator's score.
# The experiment is also saved in LangSmith: open it there to see the traces and the judges' reasons.
# This makes many Groq calls (the app, the guards and the judges), so keep the number of questions small.
results = run_evaluation("golden", "hybrid", limit=3)

for row in results:
    print("\n" + row["example"].inputs["question"])
    print("  answer:", row["run"].outputs["answer"][:150])
    for evaluation in row["evaluation_results"]["results"]:
        comment = evaluation.comment or ""
        print("  ", evaluation.key, "=", evaluation.score, "-", comment[:100])

wait_for_all_tracers()

# Release the Qdrant folder so Windows does not print an error when the script exits.
store.client.close()
