"""CV extraction module.

Extracts structured candidate information from raw resume text.
Prefers LLM-based extraction when an API key is available,
and falls back to deterministic heuristics otherwise.
"""

import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

_CV_SYSTEM_PROMPT = """You are a precise CV parser for an HR screening system.

Extract structured information from the CV text and return ONLY valid JSON — no markdown fences, no explanation, no extra text.

Return exactly this schema:
{
  "name": string or null,
  "skills": ["list of specific named technologies only — no generic terms like 'AI' or 'databases'"],
  "education": {
    "degree": string or null,
    "field": string or null,
    "institution": string or null
  },
  "experience": [
    {
      "title": string,
      "company": string,
      "type": "professional" | "internship" | "project",
      "start_year": integer or null,
      "end_year": integer or null,
      "description": string
    }
  ],
  "years_experience": float or null,
  "evidence": ["raw text snippets that contain explicit technical claims"]
}

Rules you MUST follow:
- Do NOT invent, guess, or infer missing dates. Set start_year/end_year to null if not stated.
- years_experience must only count professional roles (type = "professional"), NOT internships or projects.
- If dates are missing, set years_experience to null — do not estimate.
- Do NOT list a skill if it appears only in a negative context such as "no experience in X" or "not familiar with X".
- Do NOT treat internship experience as professional experience.
- Return null for any field that cannot be confirmed directly from the text.
"""


# --------------------------------------------------------------------------- #
# Public interface                                                              #
# --------------------------------------------------------------------------- #

def extract_structured_cv(text: str) -> dict:
    """Extract a structured candidate dict from raw CV text.

    Tries LLM-based extraction first (if API key is set). Falls back to
    heuristic extraction if the LLM is unavailable or returns invalid output.

    Args:
        text: The raw text extracted from a candidate's PDF resume.

    Returns:
        A dict with keys: name, skills, education, experience,
        years_experience, evidence.
    """
    from app.llm.client import is_llm_available, call_llm

    if is_llm_available():
        try:
            logger.info("Using LLM extraction for CV.")
            raw_response = call_llm(_CV_SYSTEM_PROMPT, text)
            return _parse_llm_json(raw_response)
        except Exception as e:
            logger.warning(f"LLM CV extraction failed ({e}). Falling back to heuristics.")

    logger.info("Using heuristic extraction for CV.")
    return _extract_heuristic(text)


# --------------------------------------------------------------------------- #
# LLM output parsing                                                           #
# --------------------------------------------------------------------------- #

def _parse_llm_json(raw: str) -> dict:
    """Parse and validate the JSON returned by the LLM.

    Args:
        raw: Raw string response from the LLM.

    Returns:
        Validated candidate dict.

    Raises:
        ValueError: If the response is not valid JSON or is missing required keys.
    """
    # Find the first { and last } to strip out <think> blocks or conversational prefixes
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        cleaned = raw[start_idx:end_idx+1]
    else:
        # Fallback if no braces found (should fail gracefully below)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

    # Normalise to the expected schema with safe defaults
    return {
        "name": data.get("name"),
        "skills": data.get("skills") or [],
        "education": data.get("education") or {},
        "experience": data.get("experience") or [],
        "years_experience": data.get("years_experience"),
        "evidence": data.get("evidence") or [],
    }


# --------------------------------------------------------------------------- #
# Heuristic fallback                                                           #
# --------------------------------------------------------------------------- #

def _extract_heuristic(text: str) -> dict:
    """Deterministic heuristic extraction from raw CV text.

    Used when LLM is unavailable. Section detection is header-based.
    Silently handles CVs that do not match the expected header names by
    returning empty lists rather than crashing.

    Args:
        text: Raw CV text.

    Returns:
        Candidate dict with the standard schema.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return _empty_candidate()

    candidate = _empty_candidate()

    # Name is typically the first line of the resume
    candidate["name"] = lines[0]

    current_section: Optional[str] = None

    section_headers = {
        "technical skills": "skills",
        "education": "education",
        "professional experience": "experience",
        "professional summary": "summary",
        "additional information": "evidence",
    }

    for line in lines:
        matched_header = False
        for header, section_key in section_headers.items():
            if line.lower().startswith(header):
                current_section = section_key
                matched_header = True
                break

        if matched_header:
            continue

        if current_section == "skills":
            skills = [s.strip() for s in line.split(",") if s.strip()]
            candidate["skills"].extend(skills)

        elif current_section == "education":
            if line:
                candidate["education"] = {"raw": line}
                # Only grab the first education line to avoid noise
                current_section = None

        elif current_section == "experience":
            candidate["experience"].append(line)

        elif current_section == "evidence":
            candidate["evidence"].append(line)

    candidate["years_experience"] = _estimate_years_heuristic(candidate["experience"])
    return candidate


def _empty_candidate() -> dict:
    """Return an empty candidate dict with the standard schema."""
    return {
        "name": None,
        "skills": [],
        "education": {},
        "experience": [],
        "years_experience": None,
        "evidence": [],
    }


def _estimate_years_heuristic(experience_lines: list) -> Optional[float]:
    """Estimate professional experience years from experience text lines.

    Internship periods are excluded from the calculation because the job
    requirement explicitly refers to professional experience.

    Args:
        experience_lines: Raw lines from the experience section.

    Returns:
        Total professional years as float, or None if no dates were found.
    """
    CURRENT_YEAR = 2026
    total_years = 0.0

    for line in experience_lines:
        # Internship periods are excluded per business rules
        if "intern" in line.lower():
            continue

        match = re.search(
            r"(\d{4})\s*(?:-|–|to)\s*(Present|\d{4})",
            line,
            re.IGNORECASE,
        )
        if match:
            start_year = int(match.group(1))
            end_val = match.group(2)
            end_year = CURRENT_YEAR if end_val.lower() == "present" else int(end_val)
            total_years += max(0, end_year - start_year)

    return total_years if total_years > 0 else None
