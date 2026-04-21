from dotenv import load_dotenv
load_dotenv()

# get the API Key
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY[:5])

# Now you need to enter the password/API key to access the models/data

# Enter this in terminal to download all the google packages

## >> uv pip install google-genai
from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # pass your api key to google to provide the models they house

# Next - Get the data from hugging_face (predominantly public presenting datasets) - # Hugging_face
# If you need pull from Kaggle, will the syntax change? Yes the calling syntax will change 

from datasets import load_dataset
import json

data = load_dataset("google/civil_comments") ["test"].select(range(5))

Abuse_types = {
    "Threat":"Explicit threats of harm, implied threats (directed profanity and obscenity).",
    "Scam":"Impersonation(posing as another person or entity), extortion-like intimidation, urgent demands or scare tactics (ex. CRA fraudulent payment requests, scam threats to citizenship)",
    "Identity based Abuse": "Slurs or demeaning language targeting protected identity groups (directed profanity and obscenity). Identity group examples include - Race/Ethnicity, Nationality/Citizenship,Religion/Belief,Sex/Gender/Gender Identity,Disability,Age, Caste/Ancestry",
    "Harassment": "Non-identity insults, humiliation, repeated hostile language",
    "Coercion": "Do X or else, manipulation, blackmail, pressure, stalking-like control. Ex. Send money now, coercive payment demands (relevant in payment memo contexts).",
    "Sexual Harassment (non-explicit)": "Unwanted sexual remarks/advances (avoid explicit content for this label).",
    "Profanity / Obscenity (non-directed)" : "Swearing not clearly targeted at a person (useful to reduce false positives).",
    "Self harm Concern" : "Sender indicates self-harm intent or ideation.",
    "Doxxing / Personal Data Exposure": "Posting someone address/phone/account info; threats to reveal such information",
    "Other Abusive": "Catch-all when clearly abusive but doesnot fit well into any category"
}

# Convert the text above into a textstring and then pass it in the prompt below
# Converting the dictionary of abuse_types and appending it into one field and passing it through the prompt
abuse_list = []
for k,v in Abuse_types.items(): 
    abuse_list.append(f"{k}:{v}")
abuse_text = "\n".join(abuse_list)           # --------- abuse_test

# prompt and create an f string. If you have a good structured prompt. Define the role, what input context, what is the comparison and instruction
# This prompt code speak to the LLM so all your instructions and data needs to be connected to this prompt f string

results = []
for item in data:
    text = item["text"]

    prompt = f"""
    Role:
    Detect if the following comments are toxic or not

    Context(input):
    {text}

    Abuse_text:
    {abuse_text}

    Instructions:
    Return only valid JSON
    No explanation before or after JSON

    Output:
    {{ "toxic": true/false,
    "abuse_type": "",
    "severity": "Low/Medium/High",
    "confidence": 0-100,
    "explanation": "" }}

    """
    
    # define the model type you wish to use
    # how did you chose this model type? Test and Trial by Ishita since the API key resulted in an error for many gemini LLMs

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )

    # text is the intial comment and the output i wish to see
    results.append({
        "text":text,
        "llm_output":response.text
    }
    )

# Save results to JSON file
with open('output.json', 'w') as f:
    json.dump(results, f, indent=4)