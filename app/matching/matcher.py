class Matcher:
    def __init__(self, weights: dict):
        self.weights = weights
        self.max_score = sum(weights.values())

    def match(self, candidate: dict) -> dict:
        score = 0
        evidence = {}
        missing = []
        
        # Combine all candidate text for simple evidence extraction
        # In a more advanced version, we'd use the structured job requirements directly.
        candidate_text_parts = []
        if candidate.get("name"): candidate_text_parts.append(candidate["name"])
        candidate_text_parts.extend(candidate.get("skills", []))
        if candidate.get("education", {}).get("raw"): candidate_text_parts.append(candidate["education"]["raw"])
        candidate_text_parts.extend(candidate.get("experience", []))
        candidate_text_parts.extend(candidate.get("evidence", []))
        
        all_text = " ".join(candidate_text_parts).lower()
        
        # 1. Education
        if any(kw in all_text for kw in ["degree", "bachelor", "master", "b.sc", "m.sc"]):
            score += self.weights.get("education", 0)
            evidence["education"] = "Found evidence of relevant degree."
        else:
            missing.append("education")
            
        # 2. Experience
        import re
        years = candidate.get("years_experience")
        exp_text = " ".join(candidate.get("experience", [])).lower()
        
        # Heuristic: only count experience if it is in a related field
        relevant_kws = [r"\bai\b", r"\bmachine learning\b", r"\bsoftware\b", r"\bdata\b", r"\bengineer\b", r"\bdeveloper\b"]
        is_relevant_exp = any(re.search(kw, exp_text) for kw in relevant_kws)
        
        if years is not None and is_relevant_exp:
            if years >= 3.0:
                score += self.weights.get("experience", 0)
                evidence["experience"] = f"Found {years} years of relevant experience."
            elif years > 0:
                partial = (years / 3.0) * self.weights.get("experience", 0)
                score += partial
                evidence["experience"] = f"Partial match: {years} years of relevant experience."
        elif years is not None and not is_relevant_exp:
            missing.append("experience")
            evidence["experience_flag"] = f"Found {years} years of experience, but it does not appear relevant to AI/ML/Software."
        else:
            missing.append("experience")
            
        # 3. Technical Skills mapping
        skill_groups = {
            "python": ["python"],
            "pytorch_tensorflow": ["pytorch", "tensorflow", "keras"],
            "nlp_llm_embeddings": ["nlp", "llm", "embedding", "rag", "transformer", "semantic search"],
            "rest_apis": ["api", "rest", "fastapi", "flask", "django", "backend"],
            "sql_postgresql": ["sql", "postgresql", "mysql", "database"],
            "docker_cloud_deploy": ["docker", "cloud", "aws", "deploy", "kubernetes"],
            "git_cicd": ["git", "ci/cd", "github actions", "gitlab"]
        }
        
        for group_key, keywords in skill_groups.items():
            weight = self.weights.get(group_key, 0)
            matched_kw = None
            
            for kw in keywords:
                found_in_positive = False
                
                # Use regex to match whole words for short keywords, or allow partial for specific ones
                # To be safe, we just pad spaces for simple checking or use regex word boundaries.
                kw_pattern = r"\b" + re.escape(kw) + r"(s|\b)" if len(kw) <= 5 else re.escape(kw)
                
                # Check normal sections
                positive_text = " ".join([candidate.get("name", "")] + candidate.get("skills", []) + [candidate.get("education", {}).get("raw", "")] + candidate.get("experience", [])).lower()
                if re.search(kw_pattern, positive_text):
                    found_in_positive = True
                else:
                    # Check evidence joined as single string to avoid split-line issues
                    evidence_text = " ".join(candidate.get("evidence", [])).lower()
                    if re.search(kw_pattern, evidence_text):
                        if "no demonstrated experience" not in evidence_text and "ambiguity flag" not in evidence_text:
                            found_in_positive = True
                                
                if found_in_positive:
                    matched_kw = kw
                    break
            
            if matched_kw:
                score += weight
                evidence[group_key] = f"Found explicit evidence of {matched_kw}."
            else:
                missing.append(group_key)
                
        # Normalize score to 100 (in case weights don't sum to exactly 100)
        final_score = (score / self.max_score) * 100 if self.max_score > 0 else 0
        
        return {
            "candidate_name": candidate.get("name", "Unknown"),
            "final_score": round(final_score, 2),
            "evidence": evidence,
            "missing": missing
        }
