"""
BHAVIK Application Entry Point
FastAPI web server serving real-time WebSockets, REST diagnostics,
and responsive mobile-first Progressive Web Application.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router
from backend.api.websocket import router as ws_router
from backend.config import FRONTEND_DIR, HOST, PORT, DEBUG

app = FastAPI(
    title="BHAVIK",
    description="Bridging Hands and Visual Intelligence through Knowledge: Real-Time ASL Fingerspelling Recognition and Transcription",
    version="1.0.0",
)

# CORS configuration for cross-device mobile testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API and WebSocket routers
app.include_router(api_router)
app.include_router(ws_router)

# PWA Root Files
@app.get("/manifest.json")
async def get_manifest():
    manifest_file = FRONTEND_DIR / "manifest.json"
    if manifest_file.exists():
        return FileResponse(manifest_file, media_type="application/manifest+json")
    return {"name": "BHAVIK", "short_name": "BHAVIK"}


@app.get("/service-worker.js")
async def get_service_worker():
    sw_file = FRONTEND_DIR / "service-worker.js"
    if sw_file.exists():
        return FileResponse(sw_file, media_type="application/javascript")
    return ""


# Mount Static Frontend Files
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting BHAVIK Server on http://{HOST}:{PORT}")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=DEBUG)
