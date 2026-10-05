# RAG Evaluation Lab

A small practice project. You ask questions about your own documents, and an AI answers using only those documents. Along the way you learn four things:

1. **Dense vs sparse search.** Two ways to find the right passage in a document, and when each one wins.
2. **Guards.** Checks that block bad questions and bad answers while the app is running.
3. **Evaluators.** Automatic tests that score the answers (is it true? is it safe? did it find the right page?).
4. **Monitoring.** Watching every request in LangSmith: how long it took, what it cost, what went wrong.

Everything is done from a web page. You only start two servers from a terminal.

## How to read these docs

Start with [docs/00-start-here.md](docs/00-start-here.md). It explains every word used in the project. Then do the phases in order. Each phase is one file, ends with a check ("you are done when..."), and gives you something you can run.

| Phase | What you build | File |
|---|---|---|
| 0 | Set up the folder, the tools and the keys | [phase-00-setup.md](docs/phase-00-setup.md) |
| 1 | Search: dense, sparse and hybrid | [phase-01-search.md](docs/phase-01-search.md) |
| 2 | Answers: the AI writes an answer with sources | [phase-02-answers.md](docs/phase-02-answers.md) |
| 3 | Tracing: see every step in LangSmith | [phase-03-tracing.md](docs/phase-03-tracing.md) |
| 4 | The backend (FastAPI) | [phase-04-api.md](docs/phase-04-api.md) |
| 5 | The web page (Streamlit) | [phase-05-ui.md](docs/phase-05-ui.md) |
| 6 | Guards and the Settings page | [phase-06-guards.md](docs/phase-06-guards.md) |
| 7 | Feedback buttons (thumbs up and down) | [phase-07-feedback.md](docs/phase-07-feedback.md) |
| 8 | Test datasets and evaluators | [phase-08-evaluators.md](docs/phase-08-evaluators.md) |
| 9 | Run evaluations from the web page | [phase-09-eval-runs.md](docs/phase-09-eval-runs.md) |
| 10 | Findings: what did you learn? | [phase-10-findings.md](docs/phase-10-findings.md) |
| 11 | Unit tests (optional practice, only at the very end) | [phase-11-tests.md](docs/phase-11-tests.md) |

