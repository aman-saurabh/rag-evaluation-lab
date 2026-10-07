import httpx
import streamlit as st
import client

TYPES = ["answerable", "unanswerable", "multi_document"]


def clean_text(value):
    """An empty cell of the table is None or nan. Turn it into an empty text."""
    if value is None:
        return ""
    text = str(value).strip()
    if text == "nan":
        return ""
    return text


def sources_to_text(sources):
    """[{"file": "a.pdf", "page": 8}, {"file": "b.pdf", "page": 9}] becomes "a.pdf:8, b.pdf:9"."""
    parts = []
    for source in sources:
        parts.append(source["file"] + ":" + str(source["page"]))
    return ", ".join(parts)


def text_to_sources(text, row_number):
    """"a.pdf:8, b.pdf:9" becomes [{"file": "a.pdf", "page": 8}, {"file": "b.pdf", "page": 9}]."""
    sources = []
    if text == "":
        return sources

    for part in text.split(","):
        file_name, colon, page = part.strip().rpartition(":")
        if colon == "" or not page.isdigit():
            raise ValueError(f"Row {row_number}: write the sources like nist-csf-2.0.pdf:8, nist-csf-2.0.pdf:9")
        sources.append({"file": file_name.strip(), "page": int(page)})
    return sources


def items_to_rows(dataset, items):
    """Makes the rows of the table. A table cell cannot hold a list, so the sources become one text."""
    rows = []
    for item in items:
        if dataset == "golden":
            rows.append({
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "answer": item["answer"],
                "sources": sources_to_text(item.get("sources", [])),
                "notes": item.get("notes", ""),
            })
        else:
            rows.append({"id": item["id"], "attack": item["attack"]})
    return rows


def rows_to_items(dataset, rows):
    """The opposite of items_to_rows: makes the list that the backend saves."""
    items = []
    for row_number, row in enumerate(rows, start=1):
        if dataset == "golden":
            item = {
                "id": clean_text(row["id"]),
                "type": clean_text(row["type"]),
                "question": clean_text(row["question"]),
                "answer": None,
            }
            answer = clean_text(row["answer"])
            if answer != "":
                item["answer"] = answer

            sources = text_to_sources(clean_text(row["sources"]), row_number)
            if len(sources) > 0:
                item["sources"] = sources

            notes = clean_text(row["notes"])
            if notes != "":
                item["notes"] = notes
        else:
            item = {"id": clean_text(row["id"]), "attack": clean_text(row["attack"])}
        items.append(item)
    return items


st.title("Test data")
st.write(
    "These are the test questions and attacks that the evaluations use (phase 9). "
    "Change a cell, add a row at the bottom, or delete rows, and press **Save** under the table. "
    "Saving rewrites the file, and comments in the file are not kept."
)

try:
    datasets = client.list_datasets()
except httpx.ConnectError:
    st.error(client.BACKEND_DOWN_MESSAGE)
    st.stop()

names = []
for dataset_info in datasets:
    names.append(dataset_info["name"])
dataset = st.selectbox("Dataset", names)

items = client.get_dataset(dataset)
rows = items_to_rows(dataset, items)

if dataset == "golden":
    st.caption(
        "type: answerable, unanswerable or multi_document. A question that is unanswerable has no answer and "
        "no sources. Sources are written like  nist-csf-2.0.pdf:8, nist-csf-2.0.pdf:9  (file:page)."
    )
    column_config = {"type": st.column_config.SelectboxColumn("type", options=TYPES, required=True)}
else:
    st.caption("Each attack is a question that tries to trick the app or to get harmful code out of it.")
    column_config = {}

edited_rows = st.data_editor(
    rows,
    num_rows="dynamic",
    column_config=column_config,
    use_container_width=True,
    key=f"editor_{dataset}",
)

if st.session_state.pop("dataset_saved", False):
    st.success("Saved.")

if st.button("Save"):
    try:
        # We gave the editor a list of rows (dictionaries), so it gives back a list of rows.
        new_items = rows_to_items(dataset, edited_rows)
        client.save_dataset(dataset, new_items)
        st.session_state.dataset_saved = True  # shown after the page reloads
        st.rerun()  # reload the page, so the table shows what was saved
    except ValueError as error:
        st.error(str(error))
    except httpx.HTTPStatusError as error:
        st.error(f"The backend did not save it: {error.response.json()['detail']}")
    except httpx.ConnectError:
        st.error(client.BACKEND_DOWN_MESSAGE)
