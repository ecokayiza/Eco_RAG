---
name: search
description: Unified Echo retrieval guidance for choosing and using search tools.
---

# Search

Use this skill to choose tools for retrieval.

## Available Tool

- Use `database_search("query", top_k=3)` for indexed project files, user-provided documents, stored notes, local knowledge, and questions that should be grounded in the database.
- You may issue multiple independent retrieval tool calls in the same provider-native response when they can safely run in parallel, but **no more than 3 calls at the same time**, and prefer 1 or 2.

## Query Discipline

- **Write focused natural-language queries with the concrete subject and needed detail**.
- **For complex questions, search clues step by step**.
- Keep result counts small: prefer `top_k=3` to `5`.
- **Iterative Query Rewriting:** If a search returns weak or unrelated evidence, do not just add more keywords. Change the search entity. For example, if searching for a "Film" fails, search for the "Director", the "Production Company", or the "Source Material" in the next turn.
- **No Redundant Queries:** When issuing parallel tool calls, each query MUST target a distinctly different semantic angle, entity, or data source. Never issue parallel searches with nearly identical keywords just to force a match.
- Stop retrieving once the evidence is enough to answer.

## Evidence Handling

- If retrieved evidence conflicts, say what differs and prefer the source that best matches the user's requested scope.
- If tools return weak evidence, explain the uncertainty instead of overstating the answer.
