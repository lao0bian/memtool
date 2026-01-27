
import os
import typer
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from memtool_core import MemoryStore, MemtoolError, init_db
from typing import Optional, Dict, Any
from datetime import datetime

# ------------------
# Setup
# ------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize MemoryStore
# Use the same environment variable as the CLI/MCP for the database path
db_path = os.environ.get("MEMTOOL_DB", "memtool.db")
store = MemoryStore(db_path=db_path)


@app.on_event("startup")
def _ensure_db_ready() -> None:
    # Ensure schema exists even when DB file already exists but uninitialized.
    init_db(db_path)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _match_filters(
    item: Dict[str, Any],
    *,
    type: Optional[str],
    key: Optional[str],
    tag: Optional[str],
    task_id: Optional[str],
    step_id: Optional[str],
    from_dt: Optional[datetime],
    to_dt: Optional[datetime],
) -> bool:
    if type and item.get("type") != type:
        return False
    if key and key not in (item.get("key") or ""):
        return False
    if tag and tag not in (item.get("tags") or []):
        return False
    if task_id is not None and item.get("task_id") != task_id:
        return False
    if step_id is not None and item.get("step_id") != step_id:
        return False

    updated_at = _parse_dt(item.get("updated_at"))
    if from_dt and updated_at and updated_at < from_dt:
        return False
    if to_dt and updated_at and updated_at > to_dt:
        return False

    return True

# ------------------
# Web Pages
# ------------------

@app.get("/")
async def read_root(request: Request):
    """Serves the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------
# REST API Endpoints
# ------------------

@app.get("/api/memories")
async def get_memories(
    type: Optional[str] = Query(None),
    key: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    task_id: Optional[str] = Query(None),
    step_id: Optional[str] = Query(None),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Lists memories with filtering and pagination."""
    from_dt = _parse_dt(from_date)
    to_dt = _parse_dt(to_date)

    # 过滤条件不完全由 core 支持，先取一批再过滤（MVP 权衡）
    all_memories = store.list(
        type=type,
        task_id=task_id,
        step_id=step_id,
        tags=[tag] if tag else None,
        limit=10000,
    )

    filtered_memories = [
        item for item in all_memories
        if _match_filters(
            item,
            type=type,
            key=key,
            tag=tag,
            task_id=task_id,
            step_id=step_id,
            from_dt=from_dt,
            to_dt=to_dt,
        )
    ]

    return filtered_memories[offset : offset + limit]


@app.get("/api/memories/{memory_id}")
async def get_memory_detail(memory_id: str):
    """Retrieves a single memory by its ID."""
    try:
        return store.get(item_id=memory_id)
    except MemtoolError as e:
        if e.payload.get("error") == "NOT_FOUND":
            return JSONResponse(status_code=404, content={"message": "Memory not found"})
        return JSONResponse(status_code=500, content=e.payload)


@app.get("/api/search")
async def search_memories(q: str = Query(..., min_length=1)):
    """Performs a full-text search on memory content."""
    return store.search(query=q)

# ------------------
# Statistics API Endpoints
# ------------------

@app.get("/api/stats/summary")
async def get_stats_summary():
    """Provides a high-level summary: total count and latest update time."""
    items = store.export_items()
    latest_dt = None
    latest_value = None
    for item in items:
        parsed = _parse_dt(item.get("updated_at"))
        if parsed and (latest_dt is None or parsed > latest_dt):
            latest_dt = parsed
            latest_value = item.get("updated_at")
    return {"total_memories": len(items), "latest_update": latest_value}

@app.get("/api/stats/timeseries")
async def get_stats_timeseries(period: str = Query("day", enum=["day", "week"])):
    """Returns memory counts over time (daily or weekly)."""
    # This logic should ideally be a direct, optimized SQL query in MemoryStore
    format_str = "%Y-%m-%d" if period == "day" else "%Y-W%W"
    memories = store.export_items()
    counts: Dict[str, int] = {}
    for mem in memories:
        updated_at = _parse_dt(mem.get("updated_at"))
        if not updated_at:
            continue
        date_key = updated_at.strftime(format_str)
        counts[date_key] = counts.get(date_key, 0) + 1
    
    # Sort by date
    sorted_counts = sorted(counts.items())
    labels = [item[0] for item in sorted_counts]
    data = [item[1] for item in sorted_counts]
    return {"labels": labels, "data": data}

@app.get("/api/stats/type-dist")
async def get_stats_type_distribution():
    """Returns the distribution of memories by type."""
    memories = store.export_items()
    counts: Dict[str, int] = {}
    for mem in memories:
        mem_type = mem.get("type") or "unknown"
        counts[mem_type] = counts.get(mem_type, 0) + 1
    return counts

@app.get("/api/stats/tag-top")
async def get_stats_top_tags(n: int = Query(10, ge=1, le=50)):
    """Returns the top N most frequent tags."""
    memories = store.export_items()
    counts: Dict[str, int] = {}
    for mem in memories:
        for tag in mem.get("tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
    sorted_tags = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"tag": tag, "count": count} for tag, count in sorted_tags[:n]]

# ------------------
# CLI to run the server
# ------------------
def _run_server(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int = typer.Option(8765, help="Port to run the server on."),
):
    """
    Launches the memtool web dashboard.
    """
    print(f"🚀 Starting memtool web server on http://{host}:{port}")
    print(f"🔗 Using database: {db_path}")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    typer.run(_run_server)


if __name__ == "__main__":
    main()
