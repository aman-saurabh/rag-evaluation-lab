import uuid
import httpx
import streamlit as st
import client

st.set_page_config(page_title="RAG Evaluation Lab")
st.title("Chat with your documents")

# st.session_state keeps values between clicks. It is empty again when you reload the browser tab.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # one id per browser tab = one conversation (memory)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = []  # the run ids of the answers that already got a thumbs up or down


# ---------- Sidebar ----------
mode = st.sidebar.selectbox("Search mode", ["dense", "sparse", "hybrid"], index=2)

# Checking the services uses a little HuggingFace credit, so only do it when you click.
st.sidebar.write("Services")
if st.sidebar.button("Check services"):
    try:
        st.session_state.health_status = client.health()
    except httpx.ConnectError:
        st.sidebar.error(client.BACKEND_DOWN_MESSAGE)

if "health_status" in st.session_state:
    for service_name, is_ok in st.session_state.health_status.items():
        if is_ok:
            st.sidebar.markdown(f"{service_name}: :green[OK]")
        else:
            st.sidebar.markdown(f"{service_name}: :red[DOWN]")


# ---------- Chat ----------
def send_feedback(run_id, score, comment):
    """Saves the thumbs up (1) or thumbs down (0) in LangSmith through the backend."""
    try:
        client.feedback(run_id, score, comment)
    except httpx.ConnectError:
        st.error(client.BACKEND_DOWN_MESSAGE)
        return
    except httpx.HTTPStatusError as error:
        st.error(f"The backend returned an error: {error.response.text}")
        return

    st.session_state.feedback_given.append(run_id)
    st.rerun()  # reload the page, so the buttons are replaced by "Thanks, saved."


def show_feedback_buttons(response):
    """Shows a thumbs up and a thumbs down under an answer. A thumbs down also asks what was wrong."""
    run_id = response["run_id"]

    if run_id in st.session_state.feedback_given:
        st.caption("Thanks, saved.")
        return

    thumbs_up_column, thumbs_down_column, _ = st.columns([1, 1, 6])  # the third column only keeps the buttons small
    if thumbs_up_column.button("👍", key=f"up_{run_id}"):
        send_feedback(run_id, 1, "")
    if thumbs_down_column.button("👎", key=f"down_{run_id}"):
        st.session_state[f"commenting_{run_id}"] = True

    # After a thumbs down, ask for a comment. (The flag keeps the box open after the page reloads.)
    if st.session_state.get(f"commenting_{run_id}", False):
        comment = st.text_input("What was wrong?", key=f"comment_{run_id}")
        if st.button("Send", key=f"send_{run_id}"):
            send_feedback(run_id, 0, comment)


def show_answer_details(response):
    """Shows the abstained badge, the guards, the sources, the trace link and the mode under an answer."""
    if response["abstained"]:
        st.badge("Abstained", color="orange")

    # A blocked request: show which guard stopped it, and why, in red.
    if response["blocked"]:
        for guard_result in response["guard_results"]:
            if guard_result["action"] == "block":
                st.error(f"Blocked by {guard_result['guard']}: {guard_result['detail']}")

    # What each guard did for this question: allow, block, drop or redact.
    with st.expander("Guards"):
        for guard_result in response["guard_results"]:
            line = f"**{guard_result['guard']}**: {guard_result['action']}"
            if guard_result["file"]:
                line = line + f" (file: {guard_result['file']}, page: {guard_result['page']})"
            if guard_result["detail"]:
                line = line + f" - {guard_result['detail']}"
            st.markdown(line)

    with st.expander("Sources"):
        if len(response["sources"]) == 0:
            st.write("No sources.")
        for source in response["sources"]:
            st.markdown(f"file: **{source['file']}**, page: **{source['page']}**")
            st.write(source["text"])

    if response["trace_url"]:
        st.markdown(f"[Open trace in LangSmith]({response['trace_url']})")

    st.caption(f"Mode: {response['mode']}")

    show_feedback_buttons(response)


# Show the earlier messages of this conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            show_answer_details(message["response"])

question = st.chat_input("Ask a question about the documents")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        error_message = ""
        with st.spinner("Thinking..."):
            try:
                response = client.ask(question, mode, st.session_state.session_id)
            except httpx.ConnectError:
                error_message = client.BACKEND_DOWN_MESSAGE
            except httpx.HTTPStatusError as error:
                error_message = f"The backend returned an error: {error.response.text}"
            except httpx.TimeoutException:
                error_message = "The backend took too long to answer. Please try again."

        # The spinner is closed here. Only now it is safe to show the error and stop the page.
        # (If st.stop() runs inside the spinner, the spinner can keep spinning.)
        if error_message:
            st.error(error_message)
            st.stop()

        st.write(response["answer"])
        show_answer_details(response)

    # Save both messages so they are shown again on the next click.
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"], "response": response}
    )
