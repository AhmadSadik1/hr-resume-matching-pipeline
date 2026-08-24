# AI HR Candidate Screening Pipeline

This project is a production-minded candidate screening pipeline designed to help recruiters process unstructured CVs against job descriptions. It automates data extraction, matching, and flagging of high-risk candidates.

## Core Features

- **Hybrid Extraction Architecture**: Utilizes Groq/OpenRouter LLMs (like Llama 3) to extract structured JSON data from unpredictable CV formats, backed by a robust Regex/Heuristic fallback engine to guarantee 100% completion even during API outages or rate limits.
- **Deterministic Matching Engine**: Calculates a transparent (0-100) match score based on explicit evidence in the CV (skills, years of experience, education). It deliberately avoids using LLMs for final scoring to prevent hallucination and ensure explainability.
- **Confidence & Flagging System**: Autonomously flags "High Risk" ambiguities—such as missing core skills, generic buzzwords, or unclear dates—reducing the system's confidence score and explicitly triggering a manual HR review.

## Architecture

- **`app/models`**: Strongly-typed dataclasses separate raw parsed data from business logic.
- **`app/llm`**: Resilient REST client utilizing the `requests` library (bypassing brittle SDK dependencies). Includes custom JSON parsing to handle reasoning models (e.g., stripping `<think>` blocks).
- **`app/extraction`**: The dual-layer extraction module (LLM-first, Regex-fallback).
- **`app/screening`**: The matching and confidence calculation engines.

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-hr-screening.git
   cd ai-hr-screening
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy the example config and add your API key:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to include your `GROQ_API_KEY`.*

## How to Run

To run the full end-to-end pipeline on the provided dataset (`data/jobs` and `data/resumes`):

```bash
python main.py
```

The system will output a formatted `HR CANDIDATE SCREENING REPORT` directly to your terminal.

## Future Enhancements
- Semantic skill matching via Vector Database (e.g., ChromaDB).
- Dynamic Technical Question Generation using LLMs based on flagged missing requirements.
- OCR integration (`pytesseract`) for scanned image PDFs.
