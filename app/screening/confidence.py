from typing import List, Dict, Tuple

def evaluate_confidence_and_flags(candidate: dict, match_result: dict, min_confidence: float = 0.7) -> Tuple[float, List[str], bool]:
    """
    Evaluates confidence score, generates flags, and determines if manual review is needed.
    
    Returns:
        confidence_score (float): 0.0 to 1.0
        flags (List[str]): List of warning messages
        manual_review (bool): True if manual review is triggered
    """
    flags = []
    confidence_penalty = 0.0
    
    # 1. Ambiguous Dates
    # If years_experience is None but they have experience entries, or if evidence mentions ambiguity
    evidence_text = " ".join(candidate.get("evidence", [])).lower()
    if candidate.get("years_experience") is None and candidate.get("experience"):
        flags.append("Ambiguous dates: Unable to calculate years of experience from CV.")
        confidence_penalty += 0.3
    elif "ambiguity flag" in evidence_text or "no clear employment dates" in evidence_text:
        flags.append("Ambiguous dates: Detected via CV extraction notes.")
        confidence_penalty += 0.3
        
    # 2. Missing Core Skills
    # E.g., Python, PyTorch/TensorFlow, NLP
    core_missing = [req for req in ["python", "pytorch_tensorflow", "nlp_llm_embeddings"] if req in match_result.get("missing", [])]
    if core_missing:
        flags.append(f"Missing core skills: {', '.join(core_missing)}")
        # Does not necessarily penalize confidence, but triggers review if many missing or is a flag
        
    # 3. Generic Skills
    # Check if the skills list is very short or uses broad terms without specifics
    skills = candidate.get("skills", [])
    if skills and all(len(s.split()) < 2 for s in skills) and len(skills) < 5:
        flags.append("Generic skills: Skill list is very brief or lacks specific technologies.")
        confidence_penalty += 0.1
    elif "described broadly" in evidence_text:
        flags.append("Generic skills: Skills described too broadly.")
        confidence_penalty += 0.15
        
    # 4. Insufficient Professional Experience
    years = candidate.get("years_experience")
    if years is not None and years < 3.0:
        flags.append(f"Insufficient professional experience: {years} years (requires 3+).")
        
    # 5. Unclear deployment evidence
    if "docker_cloud_deploy" in match_result.get("missing", []):
        flags.append("Unclear deployment evidence: Missing cloud or Docker experience.")
        
    # 6. Missing collaboration evidence
    if "git_cicd" in match_result.get("missing", []):
        flags.append("Missing collaboration evidence: No mention of Git, CI/CD, or team workflows.")
        
    # Calculate final confidence
    confidence_score = max(0.0, 1.0 - confidence_penalty)
    
    # Determine Manual Review
    # High severity flags (like ambiguous dates) or low confidence
    has_high_severity_flag = any("Ambiguous dates" in f for f in flags)
    manual_review = (confidence_score < min_confidence) or has_high_severity_flag
    
    return round(confidence_score, 2), flags, manual_review
