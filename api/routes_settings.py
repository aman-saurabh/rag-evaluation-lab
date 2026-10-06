from fastapi import APIRouter, HTTPException
from api.schemas import Settings
from app import graph
from app.guards import DEFAULT_SETTINGS, load_settings, save_settings

router = APIRouter()


@router.get("/settings", response_model=Settings)
def read_settings():
    return load_settings()


@router.put("/settings", response_model=Settings)
def update_settings(new_settings: Settings):
    # The types (true or false, numbers) are already checked by FastAPI. Here we check the names and the range.
    known_guards = list(DEFAULT_SETTINGS["guards"])
    if sorted(new_settings.guards) != sorted(known_guards):
        raise HTTPException(status_code=400, detail=f"guards must be exactly these: {known_guards}")

    known_thresholds = list(DEFAULT_SETTINGS["thresholds"])
    if sorted(new_settings.thresholds) != sorted(known_thresholds):
        raise HTTPException(status_code=400, detail=f"thresholds must be exactly these: {known_thresholds}")

    for threshold_name, value in new_settings.thresholds.items():
        if value < 0 or value > 1:
            raise HTTPException(status_code=400, detail=f"{threshold_name} must be between 0 and 1")

    settings = new_settings.model_dump()
    save_settings(settings)

    # The answer agent was built with the old settings. Build it again, so the new ones take effect.
    graph.build_answer_agent()

    return settings
