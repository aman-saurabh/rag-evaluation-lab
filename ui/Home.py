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
def show_answer_details(response):
    """Shows the abstained badge, the sources, the trace link and the mode under an answer."""
    if response["abstained"]:
        st.badge("Abstained", color="orange")

    with st.expander("Sources"):
        if len(response["sources"]) == 0:
            st.write("No sources.")
        for source in response["sources"]:
            st.markdown(f"**{source['file']}**, page {source['page']}")
            st.write(source["text"])

    if response["trace_url"]:
        st.markdown(f"[Open trace in LangSmith]({response['trace_url']})")

    st.caption(f"Mode: {response['mode']}")


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
        with st.spinner("Thinking..."):
            try:
                response = client.ask(question, mode, st.session_state.session_id)
            except httpx.ConnectError:
                st.error(client.BACKEND_DOWN_MESSAGE)
                st.stop()
            except httpx.HTTPStatusError as error:
                st.error(f"The backend returned an error: {error.response.text}")
                st.stop()
        st.write(response["answer"])
        show_answer_details(response)

    # Save both messages so they are shown again on the next click.
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"], "response": response}
    )
