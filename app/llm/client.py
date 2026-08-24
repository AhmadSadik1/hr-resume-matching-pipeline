"""LLM client wrapper for Groq.

Uses the requests library to call the Groq chat completions endpoint
directly. This avoids dependency on a specific SDK version.
"""

import os
import json
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def call_llm(system_prompt: str, user_message: str) -> str:
    """Send a prompt to the configured LLM via Groq and return the text response.

    Args:
        system_prompt: Instructions describing what the LLM should do.
        user_message: The actual content to process (e.g., raw CV text).

    Returns:
        The LLM's text response as a string.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
        RuntimeError: If the API call fails or returns an empty response.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,  # Deterministic output for structured extraction
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"LLM API request failed: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM API returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Unexpected LLM response format: {e}\nRaw: {response.text[:300]}")

    if not content:
        raise RuntimeError("LLM returned an empty response.")

    return content.strip()


def is_llm_available() -> bool:
    """Check whether an API key is configured and LLM calls can be made.

    Returns:
        True if GROQ_API_KEY is set, False otherwise.
    """
    return bool(os.getenv("GROQ_API_KEY", ""))
