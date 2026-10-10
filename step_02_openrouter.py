import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("Add OPENROUTER_API_KEY to your .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=0,
)

response = client.chat.completions.create(
    model="anthropic/claude-haiku-5.5",
    messages=[
        {
            "role": "user",
            "content": "Explain an API in one simple sentence.",
        }
    ],
    max_tokens=300,
)

print("Model:", response.model)
print("Assistant:", response.choices[0].message.content)