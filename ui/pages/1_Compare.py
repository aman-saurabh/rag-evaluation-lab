import httpx
import streamlit as st
import client

st.title("Compare the three search modes")
st.write("The same question is asked in dense, sparse and hybrid mode. This uses three answers' worth of Groq tokens.")

question = st.text_input("Question")

if st.button("Compare") and question:
    error_message = ""
    with st.spinner("Asking all three modes..."):
        try:
            result = client.compare(question)
        except httpx.ConnectError:
            error_message = client.BACKEND_DOWN_MESSAGE
        except httpx.HTTPStatusError as error:
            error_message = f"The backend returned an error: {error.response.text}"
        except httpx.TimeoutException:
            error_message = "The backend took too long to answer. Please try again."

    # The spinner is closed here. Only now it is safe to show the error and stop the page.
    if error_message:
        st.error(error_message)
        st.stop()

    modes = ["dense", "sparse", "hybrid"]
    columns = st.columns(3)
    for index in range(3):
        mode = modes[index]
        response = result[mode]
        with columns[index]:
            st.subheader(mode)
            st.write(response["answer"])
            if response["abstained"]:
                st.badge("Abstained", color="orange")
            for source in response["sources"]:
                st.caption(f"file: {source['file']}, page: {source['page']}")
