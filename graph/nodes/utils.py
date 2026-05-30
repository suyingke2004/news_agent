import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph.llm_config import get_llm


def emit_progress(stage: str, message: str):
    """Emit a progress update via LangGraph's stream writer.

    Safe to call outside of graph context (e.g., in tests) — silently no-ops.
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        writer({"stage": stage, "message": message})
    except RuntimeError:
        # Not inside a LangGraph runnable context (e.g., unit tests)
        pass


def invoke_structured(
    prompt: ChatPromptTemplate, llm_kwargs: dict, input_vars: dict
) -> str:
    """Invoke LLM with a prompt and return the raw string output."""
    llm = get_llm(**llm_kwargs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(input_vars)


def parse_json_response(raw: str) -> dict | list:
    """Parse a JSON response from LLM, handling markdown code fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())
