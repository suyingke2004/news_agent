import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """Represents a single atomic claim extracted from user input."""

    id: str = Field(description="Unique claim identifier, e.g. 'claim_0'")
    text: str = Field(description="The verifiable claim text")
    source_text: str = Field(
        default="", description="Original text this claim was extracted from"
    )


class Evidence(BaseModel):
    """Represents a piece of evidence found for a claim."""

    claim_id: str = Field(description="Which claim this evidence relates to")
    content: str = Field(description="The evidence text")
    source_url: str = Field(default="")
    source_name: str = Field(default="")
    source_type: str = Field(default="", description="news, reddit, rss, web")
    stance: Literal["supporting", "refuting", "neutral"] = Field(default="neutral")
    credibility_score: float = Field(default=0.5, description="0.0 to 1.0")


class DebateArgument(BaseModel):
    """Represents an argument from a debate agent."""

    role: Literal["advocate", "skeptic", "neutral"]
    claim_id: str
    argument: str
    key_points: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    """Represents the verdict for a single claim."""

    claim_id: str
    verdict: Literal["TRUE", "FALSE", "MIXED", "UNVERIFIED"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_evidence_count: int = 0
    refuting_evidence_count: int = 0


class FactCheckState(TypedDict, total=False):
    """State schema for the fact-checking pipeline.

    Fields with Annotated[..., operator.add] use the LangGraph reducer pattern:
    parallel nodes (via Send) accumulate their results into a single list.
    """

    # Input
    raw_input: str  # Original user input
    input_type: Literal["url", "text", "claim"]  # Detected input type
    language: str  # "zh" or "en"

    # Claim extraction
    claims: list[Claim]  # Extracted atomic claims

    # Evidence (accumulated via operator.add from parallel retrieval)
    evidence: Annotated[list[Evidence], operator.add]

    # Source credibility scores
    credibility_scores: Annotated[list[dict], operator.add]
    # Each dict: {"source_name": str, "source_url": str, "score": float, "reasons": list[str]}

    # Debate
    debate_arguments: list[DebateArgument]
    disagreement_scores: dict[str, float]  # claim_id -> disagreement score

    # Verdict
    verdicts: list[Verdict]

    # Final output
    overall_verdict: str  # TRUE / FALSE / MIXED / UNVERIFIED
    overall_confidence: float  # 0.0 to 1.0
    report_markdown: str  # Full report in markdown

    # Progress tracking (for streaming)
    current_stage: str  # Human-readable stage name

    # Errors
    errors: Annotated[list[str], operator.add]
