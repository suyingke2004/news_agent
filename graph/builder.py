from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.state import FactCheckState
from graph.nodes.claim_extractor import claim_extractor
from graph.nodes.claim_decomposer import claim_decomposer
from graph.nodes.evidence_retriever import evidence_retriever
from graph.nodes.source_credibility import source_credibility
from graph.nodes.evidence_aggregator import evidence_aggregator
from graph.nodes.multi_agent_debate import multi_agent_debate
from graph.nodes.verdict_synthesizer import verdict_synthesizer


def _route_to_evidence_retrievers(state: FactCheckState) -> list[Send]:
    claims = state.get("claims", [])
    if not claims:
        return [Send("evidence_retriever", state)]
    return [Send("evidence_retriever", state) for _ in claims]


def build_factcheck_graph() -> StateGraph:
    builder = StateGraph(FactCheckState)

    builder.add_node("claim_extractor", claim_extractor)
    builder.add_node("claim_decomposer", claim_decomposer)
    builder.add_node("evidence_retriever", evidence_retriever)
    builder.add_node("source_credibility", source_credibility)
    builder.add_node("evidence_aggregator", evidence_aggregator)
    builder.add_node("multi_agent_debate", multi_agent_debate)
    builder.add_node("verdict_synthesizer", verdict_synthesizer)

    builder.add_edge(START, "claim_extractor")
    builder.add_edge("claim_extractor", "claim_decomposer")
    builder.add_conditional_edges(
        "claim_decomposer", _route_to_evidence_retrievers, ["evidence_retriever"]
    )
    builder.add_edge("evidence_retriever", "source_credibility")
    builder.add_edge("source_credibility", "evidence_aggregator")
    builder.add_edge("evidence_aggregator", "multi_agent_debate")
    builder.add_edge("multi_agent_debate", "verdict_synthesizer")
    builder.add_edge("verdict_synthesizer", END)

    return builder


def compile_factcheck_graph():
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_factcheck_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
