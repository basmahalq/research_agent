import uuid
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import (
    build_graph,
    do_answer_from_report,
    do_evaluate,
    do_followup_check,
    do_search,
    do_write_report,
    empty_state,
    get_checkpointer,
)

app = FastAPI(title="Research Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
memory, _checkpoint_cm = get_checkpointer()
agent = build_graph().compile(checkpointer=memory)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---- full agent flow ----------------------------------------------------- #

class ResearchRequest(BaseModel):
    topic: str


class FollowupRequest(BaseModel):
    thread_id: str
    question: str


@app.post("/api/research")
def research(req: ResearchRequest):
    if not req.topic.strip():
        raise HTTPException(400, "topic is required")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(empty_state(req.topic), config=config)

    return {"thread_id": thread_id, "report": result["report"]}


@app.post("/api/followup")
def followup(req: FollowupRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    config = {"configurable": {"thread_id": req.thread_id}}
    state = agent.get_state(config)
    if not state.values:
        raise HTTPException(404, "unknown thread_id, start a new research session")

    result = agent.invoke(
        {**state.values, "query": req.question, "iterations": 0, "evaluation": ""},
        config=config,
    )
    return {"answer": result["report"]}


# ---- individual tool endpoints -------------------------------------------- #
# these expose each tool the agent uses on its own, for testing or for
# building other things on top of them.

class SearchRequest(BaseModel):
    query: str


class EvaluateRequest(BaseModel):
    results: List[str]


class WriteReportRequest(BaseModel):
    results: List[str]


class FollowupCheckRequest(BaseModel):
    report: str
    question: str


class AnswerRequest(BaseModel):
    report: str
    question: str


@app.post("/api/tools/search")
def tool_search(req: SearchRequest):
    return {"results": do_search(req.query)}


@app.post("/api/tools/evaluate")
def tool_evaluate(req: EvaluateRequest):
    return {"evaluation": do_evaluate(req.results)}


@app.post("/api/tools/write-report")
def tool_write_report(req: WriteReportRequest):
    return {"report": do_write_report(req.results)}


@app.post("/api/tools/followup-check")
def tool_followup_check(req: FollowupCheckRequest):
    return {"decision": do_followup_check(req.report, req.question)}


@app.post("/api/tools/answer")
def tool_answer(req: AnswerRequest):
    return {"answer": do_answer_from_report(req.report, req.question)}
