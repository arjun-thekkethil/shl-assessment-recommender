"""FastAPI service exposing /health, /chat, and the chat UI at /."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import SHLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_api")

_FRONTEND = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="SHL Assessment Recommender", docs_url="/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: SHLAgent | None = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    global _agent
    logger.info("Initializing SHLAgent …")
    _agent = SHLAgent()
    logger.info("SHLAgent ready")


# ---------------------------------------------------------------------------
# API endpoints  (must be declared before the catch-all static mount)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = _agent.chat(messages)

    recs = [
        Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ---------------------------------------------------------------------------
# Serve frontend — mount AFTER API routes so /health and /chat take priority
# ---------------------------------------------------------------------------

if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
