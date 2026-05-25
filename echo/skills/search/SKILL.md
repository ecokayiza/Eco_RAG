---
name: search
description: Unified Echo retrieval guidance for choosing and using search tools.
---

# Search

Use this skill to choose tools for retrieval.

## Available Tool

- Use `database_search("query", top_k=3)` for indexed project files, user-provided documents, stored notes, local knowledge, and questions that should be grounded in the database.
- You may issue multiple independent retrieval tool calls in the same provider-native response when they can safely run in parallel.

## Query Discipline

- **Write focused natural-language queries with the concrete subject and needed detail**.
- **For complex questions, search clues step by step**.
- Keep result counts small: prefer `top_k=3` to `5`.
- Avoid repeating the same search without adding a sharper term, source name, date, or constraint.
- Stop retrieving once the evidence is enough to answer.

## Evidence Handling

- If retrieved evidence conflicts, say what differs and prefer the source that best matches the user's requested scope.
- If tools return weak evidence, explain the uncertainty instead of overstating the answer.
