import json

import google.generativeai as genai

ROUTER_PROMPT = """
You are an intent classification system for a swimming pool assistant.

Your task is to classify the user's message into EXACTLY ONE of the following intents:

--------------------------------------------------
1. "qa"
--------------------------------------------------

Use this intent when:
- The user asks informational questions
- The user requests explanations
- The user asks about pool chemistry concepts
- The user asks about ideal parameter ranges
- The user asks about testing methods
- The user asks about pool maintenance

Examples:
- What is pH?
- What is the ideal chlorine range?
- Why is alkalinity important?
- How do I measure free chlorine?
- What happens if pH is too high?

--------------------------------------------------
2. "chlorine_adjustment"
--------------------------------------------------

Use this intent ONLY when:
- The user wants a chlorine dosage recommendation
- The user provides chlorine measurements
- The user asks how much chlorine to add
- The user asks how to raise chlorine levels
- The user includes pool volume and chlorine ppm

Examples:
- My chlorine is 0.5 ppm in a 15,000 gallon pool
- How much chlorine should I add?
- My free chlorine is too low
- Calculate chlorine adjustment
- I have 1 ppm chlorine in a 20k gallon pool

IMPORTANT:
- Only classify as "chlorine_adjustment"
  if the request is specifically about chlorine dosage
  or chlorine correction.

- All other informational questions must be classified as "qa".

You must ONLY return valid JSON.

Output format:

{
  "intent": "qa"
}

OR

{
  "intent": "chlorine_adjustment"
}
"""

def classify_intent(user_input: str) -> dict:

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
{ROUTER_PROMPT}

User message:
{user_input}
"""

    response = model.generate_content(prompt)

    try:

        cleaned_response = response.text.strip()

        cleaned_response = cleaned_response.replace(
            "```json",
            ""
        )

        cleaned_response = cleaned_response.replace(
            "```",
            ""
        )

        return json.loads(cleaned_response)

    except Exception:

        return {
            "intent": "qa"
        }