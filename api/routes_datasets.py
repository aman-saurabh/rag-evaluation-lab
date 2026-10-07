from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from api.schemas import AttackItem, DatasetInfo, GoldenItem
from app.eval_runner import DATASETS, load_items, save_items

router = APIRouter()


def check_dataset_name(name: str) -> None:
    """Only the three known datasets are allowed. A file path is never built from other text."""
    if name not in DATASETS:
        raise HTTPException(status_code=404, detail=f"No such dataset. Use one of {DATASETS}")


def check_item(dataset: str, item: dict, row: int) -> dict:
    """Checks one row. Returns the row as a clean dictionary, or raises an error that says which row is wrong."""
    try:
        if dataset == "golden":
            checked_item = GoldenItem(**item)
        else:
            checked_item = AttackItem(**item)
    except ValidationError as error:
        first_problem = error.errors()[0]
        field = ".".join(str(part) for part in first_problem["loc"])
        raise HTTPException(status_code=400, detail=f"Row {row}, field {field}: {first_problem['msg']}")

    if dataset == "golden":
        if checked_item.type == "unanswerable":
            if checked_item.answer is not None or len(checked_item.sources) > 0:
                raise HTTPException(status_code=400, detail=f"Row {row}: an unanswerable question has no answer and no sources")
        else:
            if checked_item.answer is None or len(checked_item.sources) == 0:
                raise HTTPException(status_code=400, detail=f"Row {row}: this question needs an answer and at least one source")

    # exclude_unset keeps only the fields that were sent, so optional fields (notes) do not appear as null.
    return checked_item.model_dump(exclude_unset=True)


@router.get("/datasets", response_model=list[DatasetInfo])
def list_datasets():
    datasets = []
    for name in DATASETS:
        datasets.append(DatasetInfo(name=name, examples=len(load_items(name))))
    return datasets


@router.get("/datasets/{name}")
def read_dataset(name: str):
    check_dataset_name(name)
    return load_items(name)


@router.put("/datasets/{name}")
def update_dataset(name: str, items: list[dict]):
    check_dataset_name(name)
    if len(items) == 0:
        raise HTTPException(status_code=400, detail="A dataset needs at least one row")

    checked_items = []
    seen_ids = []
    for row, item in enumerate(items, start=1):
        checked_item = check_item(name, item, row)
        if checked_item["id"] in seen_ids:
            raise HTTPException(status_code=400, detail=f"Row {row}: the id {checked_item['id']} is used twice")
        seen_ids.append(checked_item["id"])
        checked_items.append(checked_item)

    save_items(name, checked_items)
    return {"saved": len(checked_items)}
