"""Centralized configuration for the screening pipeline.

All thresholds, scoring weights, and environment-driven settings
live here. No business rule values should be hardcoded elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Scoring weights — must sum to 100                                            #
# --------------------------------------------------------------------------- #

SCORING_WEIGHTS: dict = {
    "education":            10,
    "experience":           20,
    "python":               10,
    "pytorch_tensorflow":   10,
    "nlp_llm_embeddings":   10,
    "rest_apis":            10,
    "sql_postgresql":       10,
    "docker_cloud_deploy":  10,
    "git_cicd":             10,
}

# --------------------------------------------------------------------------- #
# Thresholds                                                                   #
# --------------------------------------------------------------------------- #

MIN_CONFIDENCE_THRESHOLD: float = 0.7   # Below this → manual review
MIN_PROFESSIONAL_YEARS: float = 3.0     # Job requirement

# --------------------------------------------------------------------------- #
# LLM settings (Groq)                                                          #
# --------------------------------------------------------------------------- #

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
LLM_ENABLED: bool = bool(GROQ_API_KEY)
