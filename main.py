import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Get model name
model_name = os.getenv("GROQ_MODEL")

# Connect to Groq API
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Send prompt to model
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": "Explain what a RAG agent is in simple English."
        }
    ]
)

# Print AI response
print(response.choices[0].message.content)