---
name: search
description: Unified Echo retrieval guidance for choosing and using search tools. 
---

# Search

Use this skill to choose tools for retrieval.

## Tool Choice

- Use `date(timezone=None)` for the current date, current time, weekday, timezone, and questions about today, tomorrow, or yesterday.
- For current date/time/weekday/timezone questions, calling `date` is mandatory before answering. Do not answer these from model memory, training data, or an assumed internal clock.
- Use `database_search("query", top_k=3)` for indexed project files, user-provided documents, stored notes, local knowledge, and questions that should be grounded in the active Echo database.
- Use `web_search("query", max_results=5)` to find candidate public sources for fresh facts, official pages, documentation, and citations.
- Use `web_fetch("https://example.com/page", max_chars=8000)` to read a specific result when snippets are not enough to answer. When screenshot mode is enabled in Runtime Settings, `web_fetch` returns only a rendered screenshot for vision-capable models and does not extract page text.
- Call retrieval tools through the provider-native tool calling channel; never print XML tags like `<web_fetch>` or JSON tool payloads as text.
- Use database search before web search when local indexed evidence may answer the question.
- Use web search after database search when local results are empty, weak, stale, or clearly missing external context.
- Use web fetch after web search when the answer needs page-body evidence from one promising URL.
- Use both when the answer needs local project context plus external verification.
- You may issue multiple independent retrieval tool calls in the same provider-native response when they can safely run in parallel.

## Query Discipline

- **Write focused natural-language queries with the concrete subject and needed detail**.
- Keep result counts small: prefer `top_k=3` to `5` and `max_results=5` to `7`.
- Avoid repeating the same search without adding a sharper term, source name, date, or constraint.
- Stop retrieving once the evidence is enough to answer.

## Evidence Handling

- Prefer local database evidence for claims about indexed files or user-provided material.
- Prefer trustworthy web sources for external facts; cite URLs when web results inform the answer.
- If retrieved evidence conflicts, say what differs and prefer the source that best matches the user's requested scope.
- If tools return weak evidence, explain the uncertainty instead of overstating the answer.
