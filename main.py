import json
import os

from dotenv import load_dotenv
import google.generativeai as genai


def load_knowledge_base(path: str) -> dict:
    """Load the pool chemistry knowledge base"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_prompt(user_question: str, knowledge_base: dict) -> str:
    """Build the prompt sent to Gemini."""
    kb_text = json.dumps(knowledge_base, indent=2)

    prompt = f"""
        You are a pool chemistry expert assistant.

        Answer the user's question using ONLY the information provided in the knowledge base below.
        If the answer is not present in the knowledge base, say:
        "I don't have enough information in my knowledge base to answer that question."

        Knowledge Base:
        {kb_text}

        User Question:
        {user_question}
        """
    return prompt


def ask_gemini(prompt: str) -> str:
    """Send the prompt to Gemini and return the response."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text


def main():
    print("Pool Chemistry Assistant")
    print("Type 'exit' to quit.\n")

    knowledge_base = load_knowledge_base("kb.json")

    while True:
        user_question = input("You: ")

        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            prompt = build_prompt(user_question, knowledge_base)
            answer = ask_gemini(prompt)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()