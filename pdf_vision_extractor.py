import os
import base64
from typing import Optional
import pymupdf  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file automatically
load_dotenv()


def pdf_page_to_base64(page: pymupdf.Page, dpi: int = 300) -> str:
    """
    Renders a single PDF page into a high-resolution PNG image
    and returns its Base64 encoded string representation.
    """
    # 72 is standard PDF point DPI; scaling matrix ensures high clarity
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_pdf_to_txt_vision(
    pdf_path: str,
    output_txt_path: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    dpi: int = 300
) -> str:
    """
    Extracts text page-by-page from a PDF using GPT Vision with high accuracy.
    
    Args:
        pdf_path: Path to the input PDF file.
        output_txt_path: Path where extracted text will be saved.
        api_key: OpenAI API Key (defaults to OPENAI_API_KEY env variable).
        model: OpenAI Vision Model (e.g. "gpt-4o" or "gpt-4o-mini").
        dpi: Image resolution for page rendering (default 300 DPI).
        
    Returns:
        The extracted full text.
    """
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    
    system_prompt = (
        "You are an ultra-high precision document OCR system. Your goal is to transcribe ALL "
        "text from the given page image with 100% verbatim accuracy. \n"
        "Rules:\n"
        "1. Do NOT skip any word, letter, digit, punctuation mark, header, footer, table cell, or fine print.\n"
        "2. Do NOT summarize, paraphrase, correct spelling, or reformat text unless formatting preserves original layout.\n"
        "3. Preserve full structure, including tables, lists, and line ordering.\n"
        "4. If a word or number is partially blurry, examine carefully and transcribe exact content.\n"
        "5. Output ONLY the extracted verbatim raw text content without chat commentary."
    )
    
    all_pages_text = []
    
    print(f"Starting extraction for '{pdf_path}' ({total_pages} pages)...")
    
    for page_index in range(total_pages):
        page = doc[page_index]
        page_num = page_index + 1
        print(f"Processing page {page_num}/{total_pages} at {dpi} DPI...")
        
        # Render high-resolution image
        base64_image = pdf_page_to_base64(page, dpi=dpi)
        
        # Send page image to GPT Vision
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Transcribe all text from Page {page_num} verbatim without missing any details:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"  # High detail ensures full image grid processing
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,  # Zero temperature for deterministic verbatim extraction
            max_tokens=4096
        )
        
        page_text = response.choices[0].message.content or ""
        all_pages_text.append(f"--- PAGE {page_num} ---\n{page_text.strip()}\n")
        
    doc.close()
    
    full_output = "\n".join(all_pages_text)
    
    # Write result to txt file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_output)
        
    print(f"Successfully saved extracted text to '{output_txt_path}'.")
    return full_output


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_vision_extractor.py <path_to_pdf> <output_txt_path>")
        print("Ensure OPENAI_API_KEY environment variable is set.")
    else:
        pdf_input = sys.argv[1]
        txt_output = sys.argv[2]
        extract_pdf_to_txt_vision(pdf_input, txt_output)
