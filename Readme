# Research Agent

A LangGraph agent that researches a topic via web search (Tavily), drafts a report, and answers follow-up questions on the same thread — re-searching only when the follow-up actually needs new information.

## Structure

```
research_agent/
├── agent.py          # tools, Groq LLM wrapper, LangGraph graph
├── api.py            # FastAPI endpoints (full agent flow + individual tools)
├── app.py            # Gradio web interface
├── cli.py            # terminal interface
├── requirements.txt
└── .env.example
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `GROQ_API_KEY` — free key from https://console.groq.com/keys
- `TAVILY_API_KEY` — free key from https://app.tavily.com
- `DB_URI` — optional, for persistent sessions via Postgres. If unreachable, falls back to in-memory storage automatically.

## Running

**Web interface (Gradio):**
```bash
python app.py
```
Opens at `http://localhost:7860`. First message is the research topic; anything after is a follow-up on the same thread.

**API server:**
```bash
uvicorn api:app --reload
```
Runs at `http://localhost:8000`.

- `POST /api/research {topic}` → `{thread_id, report}`
- `POST /api/followup {thread_id, question}` → `{answer}`
- `POST /api/tools/search {query}` → `{results}`
- `POST /api/tools/evaluate {results}` → `{evaluation}`
- `POST /api/tools/write-report {results}` → `{report}`
- `POST /api/tools/followup-check {report, question}` → `{decision}`
- `POST /api/tools/answer {report, question}` → `{answer}`
- `GET /api/health`

**CLI:**
```bash
python cli.py
```

## Notes

- LLM: Groq API (`openai/gpt-oss-20b` by default, configurable via `GROQ_MODEL`).
- Search: Tavily, capped at 3 results per query.
- `MAX_SEARCH_ITERATIONS` (default 3) caps how many search rounds the agent will run before writing the report regardless of evaluation.