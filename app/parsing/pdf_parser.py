import pymupdf  # PyMuPDF
from pathlib import Path

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract textual content from a PDF resume using PyMuPDF.

    Args:
        pdf_path: Path to the candidate resume PDF.

    Returns:
        The extracted and normalized resume text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF contains no extractable text or is malformed.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume file not found: {pdf_path}")
        
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF file {pdf_path}. It might be malformed. Error: {e}")
        
    extracted_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text:
            # Normalize whitespace while preserving line structure
            normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
            extracted_text.append("\n".join(normalized_lines))
            
    doc.close()
    
    final_text = "\n\n".join(extracted_text).strip()
    
    if not final_text:
        raise ValueError(f"No extractable text found in {pdf_path}. OCR might be required, which is unsupported.")
        
    return final_text
