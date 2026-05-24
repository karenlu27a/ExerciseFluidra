import json
import google.generativeai as genai

EXTRACTION_PROMPT = """
You are a parameter extraction system for a pool assistant.

Extract the following information from the user message:

- current_chlorine
- pool_volume_gallons

Rules:
- current_chlorine must represent free chlorine in ppm
- pool_volume_gallons must be an integer
- If a value is missing, return null

You must ONLY return valid JSON.

Output format:
{
  "current_chlorine": 0.5,
  "pool_volume_gallons": 15000
}
"""

def calculate_chlorine_adjustment(
    current_chlorine: float,
    pool_volume_gallons: int,
    target_chlorine: float = 2.0
) -> dict:
    """
    Calculate liquid chlorine recommendation.

    Assumption:
    - 1 gallon of liquid chlorine raises chlorine
      by approximately 5 ppm in 10,000 gallons.
    """

    difference = target_chlorine - current_chlorine

    # No adjustment needed
    if difference <= 0:

        return {
            "status": "ok",
            "current_chlorine": current_chlorine,
            "target_chlorine": target_chlorine,
            "recommended_product": None,
            "estimated_amount_gallons": 0,
            "message": (
                "Your chlorine level is already within "
                "the recommended range."
            )
        }

    # Calculation
    gallons_needed = (
        difference / 5
    ) * (
        pool_volume_gallons / 10000
    )

    gallons_needed = round(gallons_needed, 2)

    return {
        "status": "adjustment_needed",
        "current_chlorine": current_chlorine,
        "target_chlorine": target_chlorine,
        "recommended_product": "Liquid chlorine",
        "estimated_amount_gallons": gallons_needed,
        "message": (
            f"Add approximately {gallons_needed} gallons "
            f"of liquid chlorine and retest after circulation."
        )
    }

def extract_chlorine_parameters(
    user_input: str
) -> dict:

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    prompt = f"""
{EXTRACTION_PROMPT}

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
            "current_chlorine": None,
            "pool_volume_gallons": None
        }


def validate_required_parameters(
    extracted_parameters: dict
) -> list[str]:

    missing_parameters = []

    if extracted_parameters.get(
        "current_chlorine"
    ) is None:

        missing_parameters.append(
            "current chlorine level (ppm)"
        )

    if extracted_parameters.get(
        "pool_volume_gallons"
    ) is None:

        missing_parameters.append(
            "pool volume in gallons"
        )

    return missing_parameters


def run_chlorine_adjustment_workflow(
    user_input: str
) -> str:

    # -----------------------------------
    # STEP 1 — Extract parameters
    # -----------------------------------
    extracted_parameters = (
        extract_chlorine_parameters(
            user_input
        )
    )

    # -----------------------------------
    # STEP 2 — Validate parameters
    # -----------------------------------
    missing_parameters = (
        validate_required_parameters(
            extracted_parameters
        )
    )

    if missing_parameters:

        missing_text = ", ".join(
            missing_parameters
        )

        return (
            f"I need additional information: "
            f"{missing_text}."
        )

    # -----------------------------------
    # STEP 3 — Run deterministic calculator
    # -----------------------------------
    calculation_result = (
        calculate_chlorine_adjustment(
            current_chlorine=float(
                extracted_parameters[
                    "current_chlorine"
                ]
            ),
            pool_volume_gallons=int(
                extracted_parameters[
                    "pool_volume_gallons"
                ]
            )
        )
    )

    # -----------------------------------
    # STEP 4 — Generate conversational response
    # -----------------------------------
    response = f"""
Pool Chlorine Recommendation

Current Chlorine:
{calculation_result['current_chlorine']} ppm

Target Chlorine:
{calculation_result['target_chlorine']} ppm

Recommended Product:
{calculation_result['recommended_product']}

Estimated Amount:
{calculation_result['estimated_amount_gallons']} gallons

Recommendation:
{calculation_result['message']}
"""

    return response.strip()