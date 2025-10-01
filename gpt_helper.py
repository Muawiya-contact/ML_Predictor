import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = API # get from .env file

def explain_prediction(prediction):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant AI."},
            {"role": "user", "content": f"The ML model predicts {prediction}. Please explain what this means for the doctor and patient."}
        ]
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print(explain_prediction("ICU"))
