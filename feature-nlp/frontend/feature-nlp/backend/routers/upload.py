"""
Upload Router — POST /upload
Accepts a CSV file, saves it, analyses schema and statistics,
generates suggested questions via LLM, and registers the dataset.
"""
import os
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

import state
from models.schemas import UploadResponse, ColumnInfo
from services.csv_analyzer import analyze_csv
from services.llm_service import suggest_questions

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", BASE_DIR / "uploads")).resolve()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    # Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    dataset_id = str(uuid.uuid4())
    dest_path = UPLOADS_DIR / f"{dataset_id}.csv"

    # Save file to disk
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Analyse CSV
    try:
        analysis = analyze_csv(str(dest_path))
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    columns = analysis["columns"]
    sample = analysis["sample"]
    row_count = analysis["row_count"]

    # Get LLM-suggested questions (non-blocking — default on failure)
    try:
        suggestions = suggest_questions(columns, sample)
    except Exception:
        suggestions = [
            "What are the top 5 records by value?",
            "Show averages grouped by category.",
            "Are there any trends over time?",
            "Which group has the highest total?",
        ]

    # Register in state
    state.datasets[dataset_id] = {
        "file_path": str(dest_path),
        "filename": file.filename,
        "row_count": row_count,
        "columns": columns,
        "sample": sample,
    }

    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        row_count=row_count,
        columns=[ColumnInfo(**c) for c in columns],
        sample=sample,
        suggested_questions=suggestions,
    )
