import os
import sys
from rostaing_ocr import ocr_extractor


def extract_pdf_to_txt(pdf_path: str, output_txt_path: str = None) -> str:
    """
    Extracts text from any PDF file using rostaing-ocr and saves it to a .txt file.
    
    Args:
        pdf_path: Path to the input PDF file.
        output_txt_path: Optional path for output text file. Defaults to same name with .txt extension.
        
    Returns:
        Path to the output .txt file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: PDF file '{pdf_path}' does not exist.")

    # If output path is not provided, generate default output filename
    if not output_txt_path:
        base_name = os.path.splitext(pdf_path)[0]
        output_txt_path = f"{base_name}_extracted.txt"

    print(f"Extracting text from '{pdf_path}'...")
    ocr_extractor(pdf_path, output_file=output_txt_path)
    print(f"Extraction complete! Text saved to '{output_txt_path}'.")
    return output_txt_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_pdf = sys.argv[1]
        output_txt = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Ask user for input file path if not passed as CLI argument
        input_pdf = input("Enter path to PDF file: ").strip().strip('"')
        output_txt = None

    if input_pdf:
        try:
            extract_pdf_to_txt(input_pdf, output_txt)
        except Exception as e:
            print(f"Extraction failed: {e}")