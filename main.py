import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

from agents.shield import validate_input
from agents.router import classify_intent
from utils.calculator_workflow import run_chlorine_adjustment_workflow
from utils.logger import logger

MAX_HISTORY = 6

def load_knowledge_base(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def format_chat_history(history: list[dict]) -> str:
    """
    Convert conversation history into prompt format.
    """

    formatted_history = ""

    for message in history:

        role = message["role"]
        content = message["content"]

        formatted_history += f"{role.upper()}: {content}\n"

    return formatted_history

def build_prompt(
    user_question: str,
    knowledge_base: dict,
    conversation_history: list[dict]
) -> str:

    kb_text = json.dumps(knowledge_base, indent=2)

    history_text = format_chat_history(conversation_history)

    prompt = f"""
You are a helpful conversational pool chemistry assistant.

Answer ONLY using the knowledge base provided below.

Use the conversation history to maintain context across turns.

If the answer is not available in the knowledge base, say:
"I don't have enough information to answer that."

====================
KNOWLEDGE BASE
====================

{kb_text}

====================
CONVERSATION HISTORY
====================

{history_text}

====================
CURRENT USER QUESTION
====================

{user_question}
"""

    return prompt

def ask_gemini(prompt: str) -> str:

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)

    return response.text

def main():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    genai.configure(api_key=api_key)

    knowledge_base = load_knowledge_base(
        "kb.json"
    )

    # -----------------------------
    # Conversation memory
    # -----------------------------
    conversation_history = []

    print("🏊 Pool Chemistry Assistant")
    print("Type 'exit' to quit.\n")

    while True:

        user_question = input("You: ")

        if user_question.lower() in {"exit", "quit"}:
            print("\nGoodbye!\n")
            break

        # -----------------------------
        # Shield validation
        # -----------------------------
        safety_result = validate_input(user_question)

        if not safety_result["is_safe"]:

            logger.warning(
            f"Unsafe query blocked | "
            f"Reason: {safety_result['reason']} | "
            f"Query: {user_question}"
)
            print(f"\n🛡️ Shield: {safety_result['reason']}\n")
            continue

        logger.info(
        f"Safe query accepted | Query={user_question}"
        )

        # -----------------------------------
        # ROUTER
        # -----------------------------------
        routing_result = classify_intent(
            user_question
        )

        intent = routing_result["intent"]

        logger.info(
            f"Intent classified | Intent={intent}"
        )
        
        if intent == "qa":

            prompt = build_prompt(
                user_question=user_question,
                knowledge_base=knowledge_base,
                conversation_history=conversation_history[-MAX_HISTORY:]
            )

            answer = ask_gemini(prompt)

        # -----------------------------
        # CHLORINE WORKFLOW
        # -----------------------------
        elif intent == "chlorine_adjustment":

            answer = (
                run_chlorine_adjustment_workflow(
                    user_question
                )
            )

        # -----------------------------
        # FALLBACK
        # -----------------------------
        else:

            answer = ("Sorry, I could not determine how to process your request.")

        # -----------------------------------
        # OUTPUT ANSWER
        # -----------------------------------
        
        print(f"\n🤖 Assistant: {answer}\n")

        logger.info(
            "Assistant response generated"
        )

        # -----------------------------------
        # SAVE MEMORY
        # -----------------------------------
        conversation_history.append({
            "role": "user",
            "content": user_question
        })

        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

if __name__ == "__main__":
    main()