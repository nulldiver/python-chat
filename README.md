# python-chat

## Features:

1. OpenAI gpt integration
2. Custom system prompting
3. Multi-threaded auto-summary of messages to fit long conversations in context length.
4. Streaming responses.

## Setup:

1. Ensure that OPENAI_API_KEY is set correctly as an environment variable 
2. Install dependencies: `pip install -r "requirements.txt"`
3. Run the test script: `python main.py`

## Example:

```
You: Hello?

Assistant: Hello! How can I assist you today?

You: Remember the word "framebunker"

Assistant: Of course! "Framebunker" is now saved in my memory. How can I help you with it?

You: Tell me a joke.

Assistant: Sure, here's a joke for you:

Why don't scientists trust atoms?

Because they make up everything!

You: What word did I ask you to remember?

Assistant: You asked me to remember the word "framebunker."
``` 