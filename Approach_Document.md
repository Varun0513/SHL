# SHL Conversational Assessment Recommender: Approach Document

## 1. Design Choices
**Stateless Architecture:** Per the requirements, the system stores zero session data. The FastAPI `/chat` endpoint accepts the entire message history on every POST request. This design enables horizontal scaling without sticky sessions and eliminates database overhead. 

**Single LLM Call via JSON Mode:** Instead of chaining multiple LLM calls (e.g., intent detection → parameter extraction → response generation) or relying on heavy agent frameworks (like LangGraph/CrewAI), the system executes a single API call to Groq's `llama-3.3-70b-versatile` model. 
* **Reasoning:** The evaluation harness enforces a strict 30-second timeout. Each API call introduces network latency and risk. A single call via JSON Mode, combined with an exhaustive system prompt, ensures the response is generated cleanly and well within the time limit.
* **Schema Enforcement:** I used the Pydantic v2 framework on top of FastAPI. The raw JSON output from the LLM is instantly validated against the exact harness schema (`ChatResponse`). If validation fails, a global exception handler gracefully returns a schema-compliant fallback rather than a 500 server crash, protecting the automated score.

## 2. Retrieval Setup
**Keyword Overlap Search over Vector Embeddings:** Instead of using a vector database (like FAISS or Chroma), the retrieval module (`CatalogSearcher`) utilizes an in-memory, exact-token overlap algorithm. 
* **Reasoning:** The SHL catalog is relatively small (~36 items) and highly domain-specific (e.g., "Verify Numerical Reasoning", "OPQ32r"). Semantic embeddings frequently over-generalize or hallucinate matches for short acronyms. A keyword-overlap algorithm guarantees deterministic retrieval, has near-zero latency (<1ms), and requires no external API calls, further safeguarding the 30-second timeout. 
* **Pre-filtering:** The system concatenates all user turns to build a query string, extracts alphanumeric tokens, and ranks the catalog. The top 15 results are dynamically injected into the system prompt. This gives the LLM enough context to answer "Compare" requests without overwhelming the context window.

## 3. Prompt Design & Guardrails
The system prompt is designed to govern 5 specific conversational behaviors:
1. **Clarify:** Returns an empty `recommendations` list if the user lacks role/seniority context.
2. **Recommend:** Instructs the LLM to select 1-10 assessments strictly from the injected catalog block.
3. **Refine:** Instructs the LLM to update its internal criteria based on mid-conversation shifts.
4. **Compare:** Forces the LLM to answer using *only* the provided catalog descriptions, avoiding prior-knowledge hallucinations.
5. **Refuse:** Redirects prompt injections and off-topic questions.

**URL Integrity Gate:** To guarantee zero URL hallucination, the prompt explicitly instructs the LLM to copy URLs exactly. However, as a fail-safe, the Python backend strips out any recommendation object whose URL does not natively exist in the loaded `catalog.json` before returning the HTTP response.

## 4. Evaluation Approach & Iterations
* **Local Test Harness:** I built a simulated 3-turn conversational python script (`test.py`) that mocks the evaluation environment, tracking endpoint latency and schema adherence.
* **What Didn't Work:** Initially, I experimented with OpenAI's `gpt-4o-mini`, but switched to Groq's `llama-3.3-70b-versatile`. Groq offered significantly lower latency, which provided a much wider safety margin against the 30-second timeout threshold. Furthermore, early iterations attempted to pass the entire catalog into the prompt. This degraded instruction adherence. Moving to the keyword-based pre-filter (Top-15) drastically improved the agent's focus.
* **Measuring Improvement:** Improvement was measured by endpoint response latency (reduced to ~1-3s) and zero occurrences of HTTP 422/500 errors during local schema-validation stress tests.

## 5. Use of AI Tools
An AI coding assistant was used heavily to accelerate boilerplate generation. It was specifically utilized for:
- Writing the Pydantic v2 schemas to ensure perfect compliance with the provided JSON shapes.
- Structuring the FastAPI boilerplate and global exception handlers.
- Drafting the initial dataset of 36 mock SHL assessments (`catalog.json`) to develop against before the final deployment. 
- Automating the migration from the OpenAI SDK to the Groq SDK.
The core logic, architectural decisions, and retrieval strategy were explicitly directed to remain minimal, explainable, and independent of abstract "black box" frameworks.
