import os
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client
from openai import OpenAI


# Load credentials from the .env file beside this script.
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)


# Outer loop: application menu.
while True:
    print("\n========== BACKEND MENU ==========")
    print("1. Ollama — local")
    print("2. OpenRouter — Claude Haiku (paid)")
    print("3. Close application")

    backend = input("Enter 1, 2, or 3: ").strip()

    if backend == "3":
        print("Application closed.")
        break

    elif backend == "1":
        client = Client(
            host="http://localhost:11434",
            timeout=120.0,
        )
        model = "qwen3:1.7b"

    elif backend == "2":
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            print("Add OPENROUTER_API_KEY to your .env file.")
            continue

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        )
        model = "anthropic/claude-haiku-5.5"

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        continue

    # Each backend selection starts a fresh conversation.
    messages = []

    print(f"\nSelected model: {model}")
    print("Type 'exit' to return to the backend menu.")

    try:
        # Inner loop: conversation with the selected backend.
        while True:
            user_question = input("\nYou: ").strip()

            if user_question.lower() == "exit":
                print("Returning to the backend menu...")
                break

            if not user_question:
                print("Please type a question.")
                continue

            request_messages = messages + [
                {
                    "role": "user",
                    "content": user_question,
                }
            ]

            try:
                if backend == "1":
                    stream = client.chat(
                        model=model,
                        messages=request_messages,
                        stream=True,
                        options={
                            "num_ctx": 4096,
                            "num_predict": 500,
                        },
                    )

                else:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=request_messages,
                        stream=True,
                        max_tokens=500,
                    )

                answer_parts = []

                print("\nAssistant: ", end="", flush=True)

                try:
                    for chunk in stream:
                        if backend == "1":
                            text = chunk.message.content or ""

                        else:
                            if getattr(chunk, "error", None):
                                raise RuntimeError(
                                    "OpenRouter stream reported an error."
                                )

                            if not chunk.choices:
                                continue

                            choice = chunk.choices[0]

                            if choice.finish_reason == "error":
                                raise RuntimeError(
                                    "OpenRouter stream failed."
                                )

                            text = choice.delta.content or ""

                        print(text, end="", flush=True)
                        answer_parts.append(text)

                finally:
                    # Release the stream even if reading it fails.
                    stream.close()

                print()

                answer = "".join(answer_parts)

                if not answer.strip():
                    raise RuntimeError(
                        "No answer text received. Try again."
                    )

                # Save only successfully completed turns.
                messages = request_messages + [
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                ]

            except Exception as error:
                print(f"\nRequest failed: {error}")
                print(
                    "This turn was not saved to conversation history."
                )

    finally:
        # Release the selected client's network resources.
        client.close()