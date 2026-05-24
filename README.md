I implemented an LLM-based safety shield before the main assistant flow.
Instead of relying exclusively on hardcoded rules, the shield uses Gemini as a lightweight classifier to detect prompt injection attempts, abusive language, and off-domain requests.

This approach improves extensibility and better reflects production conversational AI architectures.