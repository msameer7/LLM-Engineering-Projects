import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from ollama import Client
from openai import OpenAI


load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(
    page_title="Dual-Backend LLM Chat",
    page_icon="💬",
    layout="centered",
)


def clear_chat():
    st.session_state.messages = []


def stream_answer(backend, model, messages):
    if backend == "Ollama":
        with Client(
            host="http://localhost:11434",
            timeout=120.0,
        ) as client:
            stream = client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    "num_ctx": 4096,
                    "num_predict": 500,
                },
            )

            try:
                for chunk in stream:
                    text = chunk.message.content or ""
                    if text:
                        yield text
            finally:
                stream.close()

    else:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "Add OPENROUTER_API_KEY to your .env file."
            )

        with OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        ) as client:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=500,
            )

            try:
                for chunk in stream:
                    if getattr(chunk, "error", None):
                        raise RuntimeError(
                            "OpenRouter stream reported an error."
                        )

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]

                    if choice.finish_reason == "error":
                        raise RuntimeError("OpenRouter stream failed.")

                    text = choice.delta.content or ""
                    if text:
                        yield text
            finally:
                stream.close()


if "messages" not in st.session_state:
    clear_chat()

with st.sidebar:
    st.title("Chat settings")

    backend = st.selectbox(
        "Backend",
        ["Ollama", "OpenRouter"],
        key="backend",
        on_change=clear_chat,
    )

    if backend == "Ollama":
        model = "qwen3:1.7b"
        st.caption("Runs on your computer.")
    else:
        model = st.selectbox(
            "Hosted model",
            ["anthropic/claude-haiku-5.5", "openrouter/free"],
            key="hosted_model",
            on_change=clear_chat,
        )

        if model == "openrouter/free":
            st.caption("Free router; model availability and limits vary.")
        else:
            st.caption("Paid model using your OpenRouter credits.")

    st.code(model, language=None)

    st.button(
        "Clear chat",
        on_click=clear_chat,
        use_container_width=True,
    )

    st.caption("Changing backend or model starts a fresh conversation.")

st.title("💬 Dual-Backend LLM Chat")
st.caption("My first LLM engineering project • Ollama + OpenRouter")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Type your message...",
    max_chars=4000,
    submit_mode="disable",
)

if prompt and prompt.strip():
    prompt = prompt.strip()

    request_messages = st.session_state.messages + [
        {"role": "user", "content": prompt}
    ]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(backend, model, request_messages)
            )

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("No answer text received.")

            st.session_state.messages = request_messages + [
                {"role": "assistant", "content": answer}
            ]

        except Exception as error:
            st.error(f"Request failed: {error}")
            st.caption(
                "This turn was not saved. You can try again."
            )