import os
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree

langsmith_client = Client()

# Remembers the id of each project, so we ask LangSmith only once. {project name: project id}
project_ids = {}


def find_project_id() -> str:
    """The id of the LangSmith project that holds our traces. LangSmith needs it to save feedback on a trace.
    Normally this is the project from the .env file. During an evaluation, the traces go into the project of
    the experiment, so we use the project of the run that is running right now."""
    project_name = os.environ["LANGSMITH_PROJECT"]

    current_run = get_current_run_tree()
    if current_run is not None and current_run.session_name:
        project_name = current_run.session_name

    if project_name not in project_ids:
        project_ids[project_name] = langsmith_client.read_project(project_name=project_name).id
    return project_ids[project_name]
