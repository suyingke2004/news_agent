from graph.builder import compile_factcheck_graph
from graph.state import FactCheckState


class FactCheckAgent:
    def __init__(
        self,
        model_provider: str = "deepseek",
        model_name: str | None = None,
        language: str = "zh",
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.language = language
        self.graph = compile_factcheck_graph()

    def verify(self, user_input: str, thread_id: str = "default") -> dict:
        result = self.graph.invoke(
            {
                "raw_input": user_input,
                "language": self.language,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return {
            "verdict": result.get("overall_verdict", "UNVERIFIED"),
            "confidence": result.get("overall_confidence", 0.0),
            "claims": [c.model_dump() for c in result.get("claims", [])],
            "verdicts": [v.model_dump() for v in result.get("verdicts", [])],
            "report": result.get("report_markdown", ""),
        }

    def verify_stream(self, user_input: str, thread_id: str = "default"):
        for chunk in self.graph.stream(
            {
                "raw_input": user_input,
                "language": self.language,
            },
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            yield chunk
