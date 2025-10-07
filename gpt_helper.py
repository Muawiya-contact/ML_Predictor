import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("API"))

def get_gpt_explanation(prediction, patient_data):
    """
    Returns short explanation:
    - Where to admit (ICU, OT, Emergency, Resuscitation)
    - Possible disease
    - Suggested treatment
    """
    messages = [
        {"role": "system", "content": "You are a medical assistant AI. Reply in max 3 lines."},
        {"role": "user", "content": f"""
        ML model predicts: {prediction}.
        Patient data: {patient_data}.
        Provide:
        1. Admission place (ICU, OT, Emergency OR Resuscitation).
        2. Possible disease.
        3. Treatment suggestion.
        """}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=120,
        temperature=0.6
    )
    return response.choices[0].message.content.strip()
