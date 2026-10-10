# Dual-Backend LLM Chat

My first hands-on LLM engineering project, built with Python
and Streamlit.

## Features
- Ollama and OpenRouter backend selection
- Streaming responses
- Conversation history within a browser session
- Clear Chat button
- Error handling
- API credentials loaded from environment variables

## Setup
1. Install Python and Ollama.
2. Clone this repository and open its folder.
3. Create a virtual environment:
   py -m venv .venv
4. Install dependencies:
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
5. Copy .env.example to .env and add your OpenRouter API key.
6. Download the local model:
   ollama pull qwen3:1.7b
7. Keep Ollama running and launch:
   .\.venv\Scripts\python.exe -m streamlit run app.py

## Model costs
Local Ollama inference does not require a paid API.
Claude Haiku on OpenRouter requires credits.
The OpenRouter free router has availability and usage limits.

## Limitations
- History is not saved to a database.
- Switching backends or models clears the conversation.
- Automatic token-based history trimming is not implemented.
- Ollama must run on the same computer as this app.

## Learning goals
API integration, message roles, streaming, generators,
session state, credential configuration, and error handling.