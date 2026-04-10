"""LLM natural-language → QueryModel package.

See ``natural_language.py`` for the Ollama client, JSON parser, and
public ``nl_to_query_model`` entry point. The LLM is treated as
untrusted input: its JSON output is validated against the active
schema and converted to a ``QueryModel`` before any SQL is emitted.
"""
