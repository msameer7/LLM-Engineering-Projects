from ollama import Client

client = Client(host="http://localhost:11434")

messages = []

print("Chat started. Type 'exit' to quit.")

while True:
    user_question = input("\nYou: ").strip()

    if user_question.lower() == "exit":
        print("Chat ended.")
        break

    if not user_question:
        print("Please type a question.")
        continue

    messages.append(
        {"role": "user", "content": user_question}
    )

    stream = client.chat(
        model="qwen3:1.7b",
        messages=messages,
        stream=True,
    )

    print("\nAssistant: ", end="", flush=True)

    answer_parts = []

    for chunk in stream:
        text = chunk.message.content or ""

        print(text, end="", flush=True)

        answer_parts.append(text)

    answer = "".join(answer_parts)

    print()

    messages.append(
        {"role": "assistant", "content": answer}
    )