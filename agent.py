import logging
import os
from typing import List, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("research_agent")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_SEARCH_ITERATIONS = int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
DB_URI = os.getenv("DB_URI", "postgresql://admin:admin123@localhost:5432/research_agent")

groq_client = Groq(api_key=GROQ_API_KEY)
search_tool = TavilySearch(max_results=3)


def llm_invoke(prompt: str, max_tokens: int = 800) -> str:
    """Single-turn chat completion against the Groq API."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


class AgentState(TypedDict):
    query: str
    search_results: List[str]
    report: str
    iterations: int
    evaluation: str


# ---- tools -------------------------------------------------------------- #
# these are plain functions so the API can call them directly, outside the
# graph, as individual endpoints.

def do_search(query: str) -> List[str]:
    try:
        results = search_tool.invoke(query)
    except Exception as exc:
        log.error("Search failed: %s", exc)
        return []
    raw = results.get("results", []) if isinstance(results, dict) else results
    return [r.get("content", "") for r in raw if r.get("content")]


def do_evaluate(results: List[str]) -> str:
    joined = "\n".join(results)
    return llm_invoke(
        "Evaluate if these search results are sufficient to write a detailed "
        "research report. Reply with exactly one word: 'sufficient' or "
        f"'insufficient'.\n\nResults:\n{joined}",
        max_tokens=10,
    ).lower()


def do_write_report(results: List[str]) -> str:
    joined = "\n".join(results)
    return llm_invoke(f"Write a short research report based on:\n{joined}")


def do_followup_check(report: str, question: str) -> str:
    return llm_invoke(
        f"Given this research report:\n{report[:3000]}\n\n"
        f"And this follow-up question: {question}\n\n"
        "Do you need to search the web for new information to answer this? "
        "Reply with exactly one word: 'search' or 'answer'.",
        max_tokens=10,
    ).lower()


def do_answer_from_report(report: str, question: str) -> str:
    return llm_invoke(
        f"Based on this research report:\n{report[:3000]}\n\n"
        f"Answer this question concisely: {question}"
    )


# ---- graph nodes ---------------------------------------------------------- #

def search_node(state: AgentState) -> AgentState:
    log.info("Searching: %s", state["query"])
    return {
        **state,
        "search_results": do_search(state["query"]),
        "iterations": state["iterations"] + 1,
    }


def evaluate_node(state: AgentState) -> AgentState:
    return {**state, "evaluation": do_evaluate(state["search_results"])}


def should_continue(state: AgentState) -> str:
    if not state["search_results"]:
        return "write_report"
    if "sufficient" in state["evaluation"] and "insufficient" not in state["evaluation"]:
        return "write_report"
    if state["iterations"] >= MAX_SEARCH_ITERATIONS:
        return "write_report"
    return "search"


def write_report_node(state: AgentState) -> AgentState:
    return {**state, "report": do_write_report(state["search_results"])}


def route_entry(state: AgentState) -> str:
    return "search" if not state.get("report") else "handle_followup"


def handle_followup_node(state: AgentState) -> AgentState:
    return {**state, "evaluation": do_followup_check(state["report"], state["query"])}


def handle_followup_decision(state: AgentState) -> str:
    return "search" if "search" in state["evaluation"] else "answer_from_report"


def answer_from_report_node(state: AgentState) -> AgentState:
    return {**state, "report": do_answer_from_report(state["report"], state["query"])}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("search", search_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("write_report", write_report_node)
    graph.add_node("handle_followup", handle_followup_node)
    graph.add_node("answer_from_report", answer_from_report_node)

    graph.add_conditional_edges(
        START, route_entry, {"search": "search", "handle_followup": "handle_followup"}
    )
    graph.add_edge("search", "evaluate")
    graph.add_conditional_edges(
        "evaluate", should_continue, {"write_report": "write_report", "search": "search"}
    )
    graph.add_edge("write_report", END)
    graph.add_conditional_edges(
        "handle_followup",
        handle_followup_decision,
        {"search": "search", "answer_from_report": "answer_from_report"},
    )
    graph.add_edge("answer_from_report", END)

    return graph


def get_checkpointer():
    try:
        cm = PostgresSaver.from_conn_string(DB_URI)
        memory = cm.__enter__()
        memory.setup()
        log.info("Connected to Postgres for session storage.")
        return memory, cm
    except Exception as exc:
        log.warning("Postgres unavailable (%s); using in-memory storage.", exc)
        return MemorySaver(), None


def empty_state(query: str) -> AgentState:
    return {"query": query, "search_results": [], "report": "", "iterations": 0, "evaluation": ""}