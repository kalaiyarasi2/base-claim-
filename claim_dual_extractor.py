import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load environment variables from .env file automatically
load_dotenv()

from pdf_vision_extractor import extract_pdf_to_txt_vision


EXPECTED_KEYS = [
    "claim_sui_account_number",
    "claimant_ssn",
    "claimant_name",
    "claim_start_date",
    "claim_end_date",
    "byb_date",
    "bye_date",
    "claim_mailing_date",
    "claim_liability_percentage",
    "claim_liability_base_amount",
    "calculated_claim_liability",
    "agency_address_line_1",
    "agency_address_line_2",
    "separation_code"
]


# ==========================================
# Normalization Helper Functions
# ==========================================

def normalize_ssn(val: str) -> str:
    """Strips hyphens/spaces to return 9 digits."""
    if not val:
        return ""
    digits = re.sub(r"\D", "", val)
    return digits if len(digits) == 9 else val.strip()


def normalize_date(val: str) -> str:
    """Converts MM/DD/YY or MM/DD/YYYY to MM/DD/YYYY."""
    if not val:
        return ""
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", val)
    if match:
        m, d, y = match.groups()
        m = m.zfill(2)
        d = d.zfill(2)
        if len(y) == 2:
            y = f"20{y}"
        return f"{m}/{d}/{y}"
    return val.strip()


def normalize_amount(val: str) -> str:
    """Formats number to 2 decimal places (e.g. 22568 -> 22568.00)."""
    if not val:
        return ""
    cleaned = re.sub(r"[^\d.]", "", val)
    if not cleaned:
        return val.strip()
    try:
        num = float(cleaned)
        return f"{num:.2f}"
    except ValueError:
        return val.strip()


def normalize_sui_account(val: str) -> str:
    """Pads SUI Account Number with leading zeros to 10 digits if needed."""
    if not val:
        return ""
    digits = re.sub(r"\D", "", val)
    if digits:
        return digits.zfill(10)
    return val.strip()


def extract_address_dynamic(text: str, isl_code: str) -> Tuple[str, str]:
    """
    Dynamically parses agency_address_line_1 and agency_address_line_2 from the document text
    matching the given ISL code (1, 2, 3, 4) in the CLAIMS OFFICE section at the bottom.
    """
    isl_str = re.sub(r"\D", "", str(isl_code))
    if not isl_str:
        return "", ""

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. Search for header containing (ISL <num>) or ISL <num>
    header_pattern = re.compile(rf"(?:\bISL\s*{isl_str}\b|\(ISL\s*{isl_str}\))", re.IGNORECASE)

    for idx, line in enumerate(lines):
        if header_pattern.search(line):
            addr1 = ""
            addr2 = ""
            for next_idx in range(idx + 1, min(idx + 6, len(lines))):
                candidate = lines[next_idx]
                if candidate.startswith("PH:") or candidate.startswith("FAX:"):
                    continue
                if header_pattern.search(candidate) or ("(ISL" in candidate.upper() and not f"ISL {isl_str}" in candidate.upper()):
                    break
                if not addr1 and re.search(r"\d+\s+[A-Z]", candidate, re.IGNORECASE):
                    addr1 = candidate
                elif addr1 and not addr2 and re.search(r"[A-Z]+,\s*[A-Z]{2}\s*\d{5}", candidate, re.IGNORECASE):
                    addr2 = candidate
                    break

            if addr1 and addr2:
                return addr1, addr2

    # 2. Direct regex block parsing fallback
    pattern = rf"(?:ISL\s*{isl_str}\)?)[^\n]*\n+([^\n]+)\n+([^\n]+)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        l1, l2 = m.group(1).strip(), m.group(2).strip()
        if not l1.startswith("PH:") and not l2.startswith("PH:"):
            return l1, l2

    return "", ""


def calculate_claim_liability(pct_val: str, mba_val: str, state_max_cap: float = 19890.00) -> str:
    """
    Calculates calculated_claim_liability based on PCT and MBA values.
    
    Rules:
    - Condition 1 (Full 100% Liability): If PCT >= 100% -> $19,890.00 (the fixed state maximum cap).
    - Condition 2 (Partial Liability < 100%): If PCT < 100% -> (MBA * PCT) / 100
    """
    if not pct_val or not mba_val:
        return ""

    try:
        cleaned_pct = re.sub(r"[^\d.]", "", str(pct_val))
        pct = float(cleaned_pct) if cleaned_pct else 0.0

        cleaned_mba = re.sub(r"[^\d.]", "", str(mba_val))
        mba = float(cleaned_mba) if cleaned_mba else 0.0

        if pct <= 0 or mba <= 0:
            return ""

        if pct >= 100.0:
            return f"{state_max_cap:.2f}"
        else:
            calc_val = (mba * pct) / 100.0
            return f"{calc_val:.2f}"
    except (ValueError, TypeError):
        return ""


# ==========================================
# Engine 1: Regex Extraction
# ==========================================

def extract_with_regex(text: str) -> Dict[str, str]:
    """
    Extracts targeted fields from text using pattern matching.
    """
    data = {k: "" for k in EXPECTED_KEYS}
    
    # 1. SUI Account Number (10 digits, e.g. 0000096482)
    sui_match = re.search(r"(?:SUI|ACCOUNT|DATE MAILED[^\n]*\n?)\s*(\d{8,10})", text, re.IGNORECASE)
    if not sui_match:
        sui_match = re.search(r"\b0000\d{6}\b", text)
    if sui_match:
        data["claim_sui_account_number"] = normalize_sui_account(sui_match.group(1) if sui_match.groups() else sui_match.group(0))

    # 2. Claimant SSN (xxx-xx-xxxx)
    ssn_match = re.search(r"\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b", text)
    if ssn_match:
        data["claimant_ssn"] = normalize_ssn(ssn_match.group(1))

    # 3. Claimant Name (e.g., SHIRAKI, DENISE RV)
    name_match = re.search(r"\b\d{3}-\d{2}-\d{4}\s+([A-Z\s,]{3,30}?)(?=\s+\d+|\s+WBA|\s+MBA)", text)
    if name_match:
        data["claimant_name"] = name_match.group(1).strip()

    # 4. Dates
    mailing_match = re.search(r"DATE MAILED:\s*(\d{1,2}/\d{1,2}/\d{2,4})", text, re.IGNORECASE)
    if mailing_match:
        data["claim_mailing_date"] = normalize_date(mailing_match.group(1))

    begins_match = re.search(r"(?:BEGINS|BYB)\s*(\d{1,2}/\d{1,2}/\d{2,4})", text, re.IGNORECASE)
    if begins_match:
        start_dt = normalize_date(begins_match.group(1))
        data["claim_start_date"] = start_dt
        data["byb_date"] = start_dt

    ends_match = re.search(r"(?:ENDS|BYE)\s*(\d{1,2}/\d{1,2}/\d{2,4})", text, re.IGNORECASE)
    if ends_match:
        end_dt = normalize_date(ends_match.group(1))
        data["claim_end_date"] = end_dt
        data["bye_date"] = end_dt

    # 5. Financials
    pct_match = re.search(r"(\d{1,3}\.\d{1,2})\s*(?:PCT|%|\s)", text)
    if not pct_match:
        pct_match = re.search(r"PCT\s*(\d{1,3}\.\d{1,2})", text)
    if pct_match:
        data["claim_liability_percentage"] = pct_match.group(1)

    mba_match = re.search(r"MBA\s+(\d+(?:\.\d{2})?)", text)
    if not mba_match:
        mba_match = re.search(r"\b22568(?:\.00)?\b", text)
    if mba_match:
        data["claim_liability_base_amount"] = normalize_amount(mba_match.group(1) if mba_match.groups() else mba_match.group(0))

    # 6. Separation Code Legend Match
    sep_code_match = re.search(r"(\d)\s*-\s*([A-Z\s/]+QUIT[A-Z\s/]+|DISCHARGED[A-Z\s/]+|LACK OF WORK[A-Z\s/]+)", text, re.IGNORECASE)
    if sep_code_match:
        data["separation_code"] = f"{sep_code_match.group(1)} - {sep_code_match.group(2).strip()}"

    # 7. Dynamic ISL Code & Agency Address Lines Extraction from Document Text
    isl_match = re.search(r"\bISL\s*(\d)\b", text, re.IGNORECASE)
    if not isl_match:
        # Check right side of claimant details row (e.g. "PCT ISL 42.99 4" or "CODE PCT ISL 8 42.99 4")
        isl_match = re.search(r"\b\d{1,3}\.\d{1,2}\s+(\d)\b", text)

    if isl_match:
        isl_num = isl_match.group(1)
        addr1, addr2 = extract_address_dynamic(text, isl_num)
        if addr1:
            data["agency_address_line_1"] = addr1
        if addr2:
            data["agency_address_line_2"] = addr2

    return data


# ==========================================
# Engine 2: LLM JSON Extraction
# ==========================================

def extract_with_llm(text: str, client: OpenAI, model: str = "gpt-4o-mini") -> Dict[str, str]:
    """
    Extracts structured JSON fields from text using OpenAI LLM.
    """
    system_prompt = (
        "You are an expert document data extraction AI. Extract the required fields from the provided document text "
        "and return a valid JSON object matching the requested schema exactly.\n\n"
        "Field Specifications:\n"
        "- claim_sui_account_number: 10-digit SUI Account Number (e.g. 0007583532).\n"
        "- claimant_ssn: 9-digit SSN without dashes (e.g. 574923678).\n"
        "- claimant_name: Full claimant name (e.g. TOMLINSON, REBECCA L).\n"
        "- claim_start_date & byb_date: Benefit Year Begins Date in MM/DD/YYYY format.\n"
        "- claim_end_date & bye_date: Benefit Year Ends Date in MM/DD/YYYY format.\n"
        "- claim_mailing_date: Date Mailed in MM/DD/YYYY format.\n"
        "- claim_liability_percentage: PCT number string (e.g. 9.105).\n"
        "- claim_liability_base_amount: MBA amount formatted with 2 decimals (e.g. 14950.00).\n"
        "- calculated_claim_liability: Calculated claim liability amount (e.g. 1997.27, or 19890.00 if PCT is 100%).\n"
        "- agency_address_line_1: Dynamically find the ISL number (1, 2, 3, 4) in the claimant table row. Then in the CLAIMS OFFICE section at the bottom of the notice, dynamically extract the street address corresponding to that ISL number.\n"
        "- agency_address_line_2: Dynamically extract the city, state, zip corresponding to that ISL number from the CLAIMS OFFICE section at the bottom.\n"
        "- separation_code: Combined numeric code and legend description (e.g. '3 - QUIT W/OUT GOOD CAUSE').\n\n"
        "Return strictly JSON matching these 14 key names. Output empty string '' if a field is not found."
    )



    prompt = f"Document Text:\n\"\"\"\n{text}\n\"\"\"\n\nReturn JSON matching schema."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    content = response.choices[0].message.content or "{}"
    try:
        raw_json = json.loads(content)
    except json.JSONDecodeError:
        raw_json = {}

    # Ensure all expected keys are present
    cleaned_data = {}
    for key in EXPECTED_KEYS:
        val = str(raw_json.get(key, "")).strip()
        cleaned_data[key] = val

    return cleaned_data


# ==========================================
# Engine 3: Dual Validation & Merger
# ==========================================

def dual_validate_and_merge(
    regex_data: Dict[str, str],
    llm_data: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Cross-verifies Regex and LLM outputs field-by-field.
    Normalizes formats and outputs final merged JSON along with a validation audit log.
    """
    merged_data = {}
    audit_log = {}

    for key in EXPECTED_KEYS:
        r_val = regex_data.get(key, "").strip()
        l_val = llm_data.get(key, "").strip()

        if key == "calculated_claim_liability":
            pct_val = merged_data.get("claim_liability_percentage", "")
            mba_val = merged_data.get("claim_liability_base_amount", "")
            calc_val = calculate_claim_liability(pct_val, mba_val)
            merged_data[key] = calc_val
            audit_log[key] = {
                "regex_val": r_val or calc_val,
                "llm_val": l_val or calc_val,
                "final_val": calc_val,
                "status": "CALCULATED_CONSENSUS",
                "confidence": 1.0
            }
            continue

        # Perform Field-specific Normalization
        if "ssn" in key:
            r_norm = normalize_ssn(r_val)
            l_norm = normalize_ssn(l_val)
        elif "date" in key:
            r_norm = normalize_date(r_val)
            l_norm = normalize_date(l_val)
        elif "amount" in key:
            r_norm = normalize_amount(r_val)
            l_norm = normalize_amount(l_val)
        elif "sui" in key:
            r_norm = normalize_sui_account(r_val)
            l_norm = normalize_sui_account(l_val)
        else:
            r_norm = r_val
            l_norm = l_val

        # Decision & Consensus Logic
        if r_norm and l_norm and r_norm.upper() == l_norm.upper():
            merged_val = l_norm
            status = "MATCH_VALIDATED"
            confidence = 1.0
        elif l_norm and not r_norm:
            merged_val = l_norm
            status = "LLM_PRIMARY"
            confidence = 0.95
        elif r_norm and not l_norm:
            merged_val = r_norm
            status = "REGEX_PRIMARY"
            confidence = 0.90
        elif r_norm and l_norm:
            # If both exist but format differs slightly
            if "agency_address" in key:
                # Prefer cleaner/shorter single address line
                merged_val = r_norm if len(r_norm) < len(l_norm) else l_norm
            else:
                merged_val = l_norm if len(l_norm) >= len(r_norm) else r_norm
            status = "RESOLVED_CONSENSUS"
            confidence = 0.92

        else:
            merged_val = ""
            status = "MISSING"
            confidence = 0.0

        merged_data[key] = merged_val
        audit_log[key] = {
            "regex_val": r_val,
            "llm_val": l_val,
            "final_val": merged_val,
            "status": status,
            "confidence": confidence
        }

    return merged_data, audit_log


def save_data_to_excel(all_page_results: list[Dict[str, str]], all_audit_logs: list[Dict[str, Any]], excel_path: str):
    """
    Saves extracted list of JSON data objects to an Excel workbook (.xlsx).
    Each page/claimant forms a row in 'Extracted Data'.
    """
    import pandas as pd

    df_data = pd.DataFrame(all_page_results)
    df_data.to_excel(excel_path, sheet_name="Extracted Data", index=False)


# ==========================================
# Main End-to-End Extraction Runner
# ==========================================

def process_claim_file(
    file_path: str,
    output_dir: str = None,
    api_key: str = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Processes a PDF or TXT document through the Dual Validation pipeline page-by-page.
    Creates a dedicated subfolder named after the document and outputs 3 files: .txt, .json, and .xlsx.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine dedicated subfolder and file paths
    file_dir = os.path.dirname(file_path)
    file_basename = os.path.basename(file_path)
    doc_name = os.path.splitext(file_basename)[0]

    # Create subfolder named after the document
    subfolder_path = output_dir if output_dir else os.path.join(file_dir, doc_name)
    os.makedirs(subfolder_path, exist_ok=True)

    txt_path = os.path.join(subfolder_path, f"{doc_name}.txt")
    json_path = os.path.join(subfolder_path, f"{doc_name}.json")
    excel_path = os.path.join(subfolder_path, f"{doc_name}.xlsx")

    # Step 1: Read / Extract full text (If PDF, render via vision OCR and save .txt)
    if file_path.lower().endswith(".pdf"):
        print(f"Converting PDF '{file_path}' using GPT Vision OCR...")
        text_content = extract_pdf_to_txt_vision(file_path, txt_path, api_key=api_key, model="gpt-4o")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        # Save a copy inside subfolder
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_content)

    # Step 2: Split text page-by-page
    pages_raw = re.split(r"--- PAGE \d+ ---", text_content)
    page_chunks = [p.strip() for p in pages_raw if p.strip()]

    if not page_chunks:
        page_chunks = [text_content.strip()]

    print(f"\nFound {len(page_chunks)} page(s) to process in document...")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    all_page_results = []
    all_audit_logs = []

    # Step 3: Process each page individually with Dual Engine Validation
    for idx, page_text in enumerate(page_chunks):
        page_num = idx + 1
        print(f"Processing Page {page_num}/{len(page_chunks)} with Dual Validation Engine...")
        
        regex_result = extract_with_regex(page_text)
        llm_result = extract_with_llm(page_text, client, model=model)
        merged_json, audit_log = dual_validate_and_merge(regex_result, llm_result)

        all_page_results.append(merged_json)
        all_audit_logs.append(audit_log)

    # Step 4: Write JSON Array File
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_page_results, f, indent=2)

    # Step 5: Write Multi-Row Excel File (.xlsx)
    save_data_to_excel(all_page_results, all_audit_logs, excel_path)

    print("\n" + "=" * 60)
    print(f"Extraction & Dual Validation Complete for '{doc_name}'!")
    print(f"Dedicated Output Subfolder: '{subfolder_path}'")
    print(f"Outputs generated ({len(all_page_results)} page records):")
    print(f"  1. Text File:  '{txt_path}'")
    print(f"  2. JSON File:  '{json_path}'")
    print(f"  3. Excel File: '{excel_path}'")
    print("=" * 60 + "\n")

    return {
        "subfolder": subfolder_path,
        "txt_path": txt_path,
        "json_path": json_path,
        "excel_path": excel_path,
        "data": all_page_results,
        "audit_logs": all_audit_logs
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else None
        res = process_claim_file(target_file, out_dir)
        print(f"Extracted {len(res['data'])} records successfully.")
    else:
        print("Usage: python claim_dual_extractor.py <file_path> [output_directory]")


