# SHL Assessment Recommender — Quick Start

## 1. Setup
```bash
# Clone / copy files to your project folder
cd SHL

# Create your .env file
copy .env.example .env
# Open .env and paste your Groq API key

# Install dependencies
pip install -r requirements.txt
```

## 2. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 3. Test the endpoints

### Health check
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I am hiring a Java developer\"}]}"
```

Interactive API docs: http://localhost:8000/docs

---

## File Overview

| File | Role |
|---|---|
| `catalog.json` | Ground-truth SHL Individual Test Solutions (36 items) |
| `catalog.py` | `CatalogSearcher` — keyword search + URL integrity |
| `schemas.py` | Pydantic v2 request/response contracts |
| `agent.py` | Keyword pre-filter → single LLM call → parse + validate |
| `main.py` | FastAPI app with `/health` and `/chat` |

## Architecture (Interview Summary)

```
POST /chat
  │
  ├─ 1. Extract keywords from user messages   [pure Python, 0ms]
  ├─ 2. CatalogSearcher.search(query, top_k=15) [pure Python, <1ms]
  ├─ 3. Render system prompt with catalog block  [pure Python, 0ms]
  ├─ 4. Groq llama-3.3-70b-versatile (JSON mode, temp=0.2) [~1-3s network]
  ├─ 5. json.loads() + Pydantic validation        [<1ms]
  └─ 6. URL integrity gate → return ChatResponse  [<1ms]
```

**Total budget:** well under the 30-second harness timeout.
