import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def generate_text(system_prompt: str, user_prompt: str, history_messages: list[dict] | None = None) -> str:
    """
    Generate a response using OpenAI chat completions.
    system_prompt: overall instruction
    user_prompt: task prompt (already includes context/report)
    history_messages: optional list of prior messages (role/content)
    """
    messages = [{"role": "system", "content": system_prompt}]

    if history_messages:
        messages.extend(history_messages)

    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
