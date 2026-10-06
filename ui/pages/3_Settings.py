import httpx
import streamlit as st
import client

GUARD_DESCRIPTIONS = {
    "length": "Stops an empty question, or one longer than 1000 characters.",
    "prompt_injection_input": "Stops a question that tries to trick the AI (for example \"ignore your instructions\").",
    "safety_input": "Stops a rude question, or one that asks for harmful commands or code.",
    "prompt_injection_document": "Throws away a retrieved document that contains instructions for the AI. The question carries on without it.",
    "pii_secrets_output": "Hides private data in the answer: emails, card numbers, IP addresses, API keys and passwords.",
    "hallucination_output": "Withholds an answer that says things the retrieved documents do not support (a made-up answer). A judge model reads the answer and the documents.",
    "safety_output": "Withholds an answer that is rude or contains harmful code.",
    "citation_output": "Withholds an answer that cites a source that does not exist, or cites no source at all.",
}

THRESHOLD_DESCRIPTIONS = {
    "dense_min_score": "Dense search: if the best match scores below this, the app says \"I don't know\".",
    "prompt_guard_min_score": "Prompt Guard: a question or document scoring at or above this counts as a trick.",
}

st.title("Settings")
st.write(
    "Switch each guard on or off and change the numbers they use. Every guard that is on adds a little time, "
    "and the safety guards use Groq tokens. Turn a guard off, send an attack on the Chat page, and watch it get through."
)
st.info("Your changes are only used after you press **Save** at the bottom of this page.")

# After a successful save the page reloads, and the message is shown here.
if st.session_state.pop("settings_saved", False):
    st.success("Saved. The new settings are used from the next question.")

try:
    settings = client.get_settings()
except httpx.ConnectError:
    st.error(client.BACKEND_DOWN_MESSAGE)
    st.stop()

# ---------- Guards ----------
st.subheader("Guards")
new_guards = {}
for guard_name, guard_setting in settings["guards"].items():
    is_on = st.toggle(guard_name, value=guard_setting["enabled"])
    st.caption(GUARD_DESCRIPTIONS.get(guard_name, ""))
    new_guards[guard_name] = {"enabled": is_on}

# ---------- Numbers ----------
st.subheader("Numbers")
new_thresholds = {}
for threshold_name, value in settings["thresholds"].items():
    new_thresholds[threshold_name] = st.number_input(
        threshold_name, min_value=0.0, max_value=1.0, value=float(value), step=0.01
    )
    st.caption(THRESHOLD_DESCRIPTIONS.get(threshold_name, ""))

# ---------- Save ----------
# Warn when what you see on the page is different from what is saved.
if new_guards != settings["guards"] or new_thresholds != settings["thresholds"]:
    st.warning("You have changes that are not saved yet. Press Save.")

if st.button("Save"):
    try:
        client.save_settings({"guards": new_guards, "thresholds": new_thresholds})
        st.session_state.settings_saved = True  # shown at the top after the page reloads
        st.rerun()  # reload the page, so it shows the saved settings and no "not saved" warning
    except httpx.ConnectError:
        st.error(client.BACKEND_DOWN_MESSAGE)
    except httpx.HTTPStatusError as error:
        st.error(f"The backend returned an error: {error.response.text}")
