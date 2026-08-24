"""Job description extraction module.

Extracts structured requirements from raw job description text.
Prefers LLM-based extraction when an API key is available,
and falls back to deterministic heuristics otherwise.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

_JD_SYSTEM_PROMPT = """You are a precise job description parser for an HR screening system.

Extract structured requirements from the job description text and return ONLY valid JSON — no markdown fences, no explanation, no extra text.

Return exactly this schema:
{
  "education": [
    { "requirement": string, "mandatory": true | false }
  ],
  "experience": [
    { "requirement": string, "mandatory": true | false, "min_years": float or null }
  ],
  "technical_skills": [
    { "requirement": string, "mandatory": true | false }
  ],
  "backend": [
    { "requirement": string, "mandatory": true | false }
  ],
  "deployment": [
    { "requirement": string, "mandatory": true | false }
  ],
  "soft_skills": [
    { "requirement": string, "mandatory": true | false }
  ]
}

Rules:
- Set mandatory = true for items listed under "Required Qualifications".
- Set mandatory = false for items listed under "Preferred", "Nice to have", or "Bonus".
- Extract min_years as a float from experience requirements (e.g. "3+ years" → 3.0).
- Categorize each bullet into the most appropriate group.
"""

# --------------------------------------------------------------------------- #
# Public interface                                                              #
# --------------------------------------------------------------------------- #

def extract_job_requirements(text: str) -> dict:
    """Extract structured requirements from raw job description text.

    Tries LLM-based extraction first (if API key is set). Falls back to
    heuristic extraction if the LLM is unavailable or returns invalid output.

    Args:
        text: The raw text extracted from a job description PDF.

    Returns:
        A dict with keys: education, experience, technical_skills, backend,
        deployment, soft_skills. Each value is a list of requirement dicts
        containing at least: requirement (str), mandatory (bool).
    """
    from app.llm.client import is_llm_available, call_llm

    if is_llm_available():
        try:
            logger.info("Using LLM extraction for job description.")
            raw_response = call_llm(_JD_SYSTEM_PROMPT, text)
            return _parse_llm_json(raw_response)
        except Exception as e:
            logger.warning(f"LLM JD extraction failed ({e}). Falling back to heuristics.")

    logger.info("Using heuristic extraction for job description.")
    return _extract_heuristic(text)


# --------------------------------------------------------------------------- #
# LLM output parsing                                                           #
# --------------------------------------------------------------------------- #

def _parse_llm_json(raw: str) -> dict:
    """Parse and validate the JSON returned by the LLM.

    Args:
        raw: Raw string response from the LLM.

    Returns:
        Validated requirements dict.

    Raises:
        ValueError: If the response is not valid JSON.
    """
    # Find the first { and last } to strip out <think> blocks or conversational prefixes
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        cleaned = raw[start_idx:end_idx+1]
    else:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

    # Normalise with safe defaults
    return {
        "education": data.get("education") or [],
        "experience": data.get("experience") or [],
        "technical_skills": data.get("technical_skills") or [],
        "backend": data.get("backend") or [],
        "deployment": data.get("deployment") or [],
        "soft_skills": data.get("soft_skills") or [],
    }


# --------------------------------------------------------------------------- #
# Heuristic fallback                                                           #
# --------------------------------------------------------------------------- #

def _extract_heuristic(text: str) -> dict:
    """Deterministic heuristic extraction from raw job description text.

    Parses the 'Required Qualifications' section and categorises each
    bullet point into the appropriate requirement group.

    Args:
        text: Raw job description text.

    Returns:
        Requirements dict with the standard schema.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    requirements: dict = {
        "education": [],
        "experience": [],
        "technical_skills": [],
        "backend": [],
        "deployment": [],
        "soft_skills": [],
    }

    in_required_section = False
    current_req = ""

    for line in lines:
        if line.lower().startswith("required qualifications"):
            in_required_section = True
            continue
        elif in_required_section and line.lower().startswith("screening notes"):
            if current_req:
                _categorize_and_add(current_req, requirements)
            in_required_section = False
            continue

        if in_required_section:
            if re.match(r"^[\W_]", line):
                if current_req:
                    _categorize_and_add(current_req, requirements)
                current_req = re.sub(r"^[\W_]+", "", line).strip()
            else:
                # Continuation of a wrapped line
                current_req += " " + line.strip()

    if current_req and in_required_section:
        _categorize_and_add(current_req, requirements)

    return requirements


def _categorize_and_add(req_text: str, requirements: dict) -> None:
    """Categorise a single requirement string and append it to the correct group.

    Args:
        req_text: The requirement text, already cleaned of bullet markers.
        requirements: The mutable requirements dict to update in place.
    """
    req_item = {"requirement": req_text, "mandatory": True}
    line_lower = req_text.lower()

    if "degree" in line_lower or "bachelor" in line_lower or "master" in line_lower:
        requirements["education"].append(req_item)
    elif "years of professional experience" in line_lower:
        match = re.search(r"(\d+)\+", line_lower)
        if match:
            req_item["min_years"] = float(match.group(1))
        requirements["experience"].append(req_item)
    elif "python" in line_lower or "pytorch" in line_lower or "nlp" in line_lower or "llm" in line_lower:
        requirements["technical_skills"].append(req_item)
    elif "api" in line_lower or "backend" in line_lower or "sql" in line_lower or "postgresql" in line_lower:
        requirements["backend"].append(req_item)
    elif "docker" in line_lower or "cloud" in line_lower or "ci/cd" in line_lower or "deployment" in line_lower:
        requirements["deployment"].append(req_item)
    elif "communication" in line_lower or "teamwork" in line_lower or "problem-solving" in line_lower:
        requirements["soft_skills"].append(req_item)
    else:
        requirements["technical_skills"].append(req_item)
