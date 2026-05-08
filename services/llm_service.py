import os
from http import client
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def run_agent(system_prompt, user_prompt, max_tokens=1000):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()
