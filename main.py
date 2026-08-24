import json
from pathlib import Path
import logging

from app.parsing.pdf_parser import extract_pdf_text
from app.extraction.cv_extractor import extract_structured_cv
from app.extraction.job_extractor import extract_job_requirements
from app.matching.matcher import Matcher
from app.screening.confidence import evaluate_confidence_and_flags
from app.config import SCORING_WEIGHTS

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    data_dir = Path("data")
    jobs_dir = data_dir / "jobs"
    resumes_dir = data_dir / "resumes"

    # 1. Process Job Description
    jd_files = list(jobs_dir.glob("*.pdf"))
    if not jd_files:
        logger.error(f"No job description PDFs found in {jobs_dir}")
        return

    # Use the first JD for this pipeline run
    jd_path = jd_files[0]
    logger.info(f"Processing Job Description: {jd_path.name}")
    try:
        jd_text = extract_pdf_text(jd_path)
        jd_requirements = extract_job_requirements(jd_text)
        logger.info("Successfully extracted structured job requirements.")
    except Exception as e:
        logger.error(f"Failed to process job description: {e}")
        return

    # Initialize the matching engine
    matcher = Matcher(SCORING_WEIGHTS)

    # 2. Process Candidates
    resume_files = list(resumes_dir.glob("*.pdf"))
    if not resume_files:
        logger.warning(f"No resume PDFs found in {resumes_dir}")
        return

    print("\n" + "="*50)
    print(" HR CANDIDATE SCREENING REPORT ".center(50, "="))
    print("="*50 + "\n")

    results = []

    for resume_path in resume_files:
        logger.info(f"Screening candidate: {resume_path.name}")
        
        try:
            # Step A: Parse PDF to raw text
            raw_cv_text = extract_pdf_text(resume_path)
            
            # Step B: Extract structured data (LLM or heuristic)
            candidate_data = extract_structured_cv(raw_cv_text)
            
            # Step C: Match against job requirements
            match_result = matcher.match(candidate_data)
            
            # Step D: Evaluate confidence and flags
            conf_score, flags, manual_review = evaluate_confidence_and_flags(
                candidate=candidate_data, 
                match_result=match_result
            )
            
            # Compile result
            candidate_result = {
                "file_name": resume_path.name,
                "candidate_name": match_result.get("candidate_name", "Unknown"),
                "match_score": match_result.get("final_score", 0.0),
                "confidence_score": conf_score,
                "manual_review_required": manual_review,
                "flags": flags,
                "missing_skills": match_result.get("missing", [])
            }
            results.append(candidate_result)
            
        except Exception as e:
            logger.error(f"Failed to process {resume_path.name}: {e}")

    # Sort results by match score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)

    # Print Report
    for idx, res in enumerate(results, 1):
        print(f"[{idx}] {res['candidate_name']} ({res['file_name']})")
        print(f"    Match Score: {res['match_score']}/100")
        print(f"    Confidence : {res['confidence_score'] * 100}%")
        
        if res['manual_review_required']:
            print("    [!] MANUAL REVIEW REQUIRED [!]")
            
        if res['flags']:
            print("    Flags:")
            for flag in res['flags']:
                print(f"      - {flag}")
                
        if res['missing_skills']:
            print("    Missing Requirements:")
            print(f"      - {', '.join(res['missing_skills'])}")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
