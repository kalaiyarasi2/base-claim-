import os
import shutil
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import process_screenshot_file, EXPECTED_KEYS
from token_monitor import get_cumulative_metrics

app = FastAPI(
    title="Unemployment Claim Extraction & Validation API",
    description="""
### Dual Vision Extraction, Field Validation & Token Cost Monitor

This API extracts data from dual-panel screenshots (PDF Notice on left vs UI Entry Screen on right), compares values field-by-field, generates Excel validation reports, and tracks exact token consumption & USD cost per case.

* **Interactive Swagger Documentation**: Automatically available at `/docs`.
* **Token Cost Monitoring**: Track per-file prompt/completion tokens and estimated USD cost.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for browser integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PathExtractionRequest(BaseModel):
    image_path: str
    output_dir: Optional[str] = None

class FolderBatchRequest(BaseModel):
    folder_path: str
    output_dir: Optional[str] = None


@app.get("/", include_in_schema=False)
def root():
    """Redirect root path to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.post(
    "/api/extract-screenshot",
    summary="Upload & Process Screenshot Image",
    tags=["Extraction & Validation"]
)
async def extract_screenshot_file(
    file: UploadFile = File(..., description="Screenshot image file (.png, .jpg, .jpeg)"),
    output_dir: Optional[str] = None
):
    """
    **Upload a split-screen screenshot image file**.
    - Extracts `pdf` fields from left panel & `ocr` fields from right panel.
    - Compares values field-by-field (`Match`/`Mismatch`).
    - Tracks prompt/completion token usage & USD cost per file.
    - Generates 5-column Excel validation spreadsheet.
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only image files (.png, .jpg, .jpeg) are supported.")

    temp_dir = tempfile.mkdtemp()
    temp_img_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        target_out_dir = output_dir or os.path.join("output", os.path.splitext(file.filename)[0])
        result = process_screenshot_file(temp_img_path, target_out_dir)
        
        # Add Excel download link
        case_name = os.path.splitext(file.filename)[0]
        result["excel_download_url"] = f"/api/download-excel/{case_name}"

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary upload copy
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post(
    "/api/extract-path",
    summary="Process Screenshot by File Path",
    tags=["Extraction & Validation"]
)
def extract_by_path(req: PathExtractionRequest):
    """Process a screenshot image by specifying its local file path."""
    abs_path = os.path.abspath(req.image_path)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"File not found at path: {abs_path}")

    try:
        result = process_screenshot_file(abs_path, req.output_dir)
        case_name = os.path.splitext(os.path.basename(abs_path))[0]
        result["excel_download_url"] = f"/api/download-excel/{case_name}"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/batch-extract-folder",
    summary="Batch Process Directory of Screenshots",
    tags=["Batch Processing"]
)
def batch_extract_folder(req: FolderBatchRequest):
    """Batch process all screenshot images inside a directory and return detailed cost breakdown."""
    abs_folder = os.path.abspath(req.folder_path)
    if not os.path.exists(abs_folder) or not os.path.isdir(abs_folder):
        raise HTTPException(status_code=404, detail=f"Folder not found at path: {abs_folder}")

    image_extensions = (".png", ".jpg", ".jpeg")
    processed_cases = []
    total_batch_cost = 0.0
    total_batch_tokens = 0

    for fname in os.listdir(abs_folder):
        if fname.lower().endswith(image_extensions):
            img_path = os.path.join(abs_folder, fname)
            try:
                res = process_screenshot_file(img_path, req.output_dir)
                token_info = res.get("token_usage", {})
                total_batch_cost += token_info.get("estimated_cost_usd", 0.0)
                total_batch_tokens += token_info.get("total_tokens", 0)

                case_name = os.path.splitext(fname)[0]
                processed_cases.append({
                    "filename": fname,
                    "case_name": case_name,
                    "validation_summary": res.get("validation", {}),
                    "token_usage": token_info,
                    "excel_download_url": f"/api/download-excel/{case_name}"
                })
            except Exception as e:
                processed_cases.append({"filename": fname, "error": str(e)})

    return {
        "status": "success",
        "processed_count": len(processed_cases),
        "total_batch_tokens": total_batch_tokens,
        "total_batch_cost_usd": round(total_batch_cost, 6),
        "formatted_batch_cost": f"${total_batch_cost:.6f}",
        "cases": processed_cases
    }


@app.get(
    "/api/token-metrics",
    summary="Get Token Consumption & USD Cost Metrics",
    tags=["Analytics & Monitoring"]
)
def get_token_metrics():
    """Returns cumulative token metrics, per-case cost breakdown, and total spent ($ USD)."""
    return get_cumulative_metrics()


@app.get(
    "/api/download-excel/{case_name}",
    summary="Download 5-Column Excel Validation Report",
    tags=["Reports"]
)
def download_excel(case_name: str):
    """Downloads the generated 5-column Excel validation spreadsheet (.xlsx) for a case."""
    # Check default output folder locations
    possible_paths = [
        os.path.join("output", case_name, f"{case_name}.xlsx"),
        os.path.join("output", case_name, f"{case_name}_updated.xlsx"),
        os.path.join("Unemployment Claims - Automation Project", "output", case_name, f"{case_name}.xlsx"),
        os.path.join("Unemployment Claims - Automation Project", "output", case_name, f"{case_name}_updated.xlsx")
    ]

    for excel_path in possible_paths:
        if os.path.exists(excel_path):
            return FileResponse(
                path=excel_path,
                filename=os.path.basename(excel_path),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    raise HTTPException(status_code=404, detail=f"Excel validation report not found for case: '{case_name}'")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
