"""
FastAPI application entry point.
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _load_environment() -> None:
    """Load .env files from current and ancestor directories."""
    this_file = Path(__file__).resolve()
    search_dirs = [this_file.parent, *this_file.parents]
    for directory in search_dirs:
        for filename in (".env.local", ".env"):
            candidate = directory / filename
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=False)


_load_environment()

from nlp2sql import generate_sql_query
from routers import upload, query, data_health_router, auto_visualize, correlation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nlp2sql-api")


class GenerateSQLColumn(BaseModel):
    name: str = Field(..., min_length=1)
    type: Optional[str] = None


class GenerateSQLRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    columns: Optional[List[GenerateSQLColumn]] = None


class GenerateSQLResponse(BaseModel):
    sql: str

app = FastAPI(
    title="Talk to Data API",
    description="CSV analytics API with local NLP2SQL generation endpoint",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Always allow local dev servers. Add more origins via FRONTEND_URL if needed.
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:3001",
]

# Allow overriding / adding more origins via env var (comma-separated)
_extra = os.environ.get("FRONTEND_URL", "")
if _extra:
    for _url in _extra.split(","):
        _url = _url.strip().rstrip("/")
        if _url and _url not in _allowed_origins:
            _allowed_origins.append(_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["Upload"])
app.include_router(query.router, tags=["Query"])
app.include_router(data_health_router.router, tags=["Data Health"])
app.include_router(auto_visualize.router, tags=["Auto Visualize"])
app.include_router(correlation.router, tags=["Correlation"])


@app.post("/generate-sql", response_model=GenerateSQLResponse, tags=["NLP2SQL"])
async def generate_sql(payload: GenerateSQLRequest) -> GenerateSQLResponse:
    """Generate SQL from natural language using local NLP2SQL logic only."""
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    schema_columns = None
    if payload.columns:
        schema_columns = [column.model_dump(exclude_none=True) for column in payload.columns]

    try:
        sql = generate_sql_query(query_text, schema_columns=schema_columns)
    except ValueError as exc:
        logger.warning("NLP2SQL validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected NLP2SQL error")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate SQL due to an internal error.",
        ) from exc

    return GenerateSQLResponse(sql=sql)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Talk to Data API is running."}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
