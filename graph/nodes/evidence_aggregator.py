import hashlib

from langchain_core.prompts import ChatPromptTemplate

from graph.state import FactCheckState, Evidence
from graph.nodes import utils as node_utils


def _content_hash(content: str, source_url: str) -> str:
    """Create a deterministic hash for deduplication."""
    normalized = f"{content.strip().lower()}|{source_url.strip().lower()}"
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _build_stance_prompt(language: str) -> ChatPromptTemplate:
    if language == "zh":
        system = "你是一个立场分类专家。判断每条证据对给定声明是支持、反驳还是中立。\n\n请以JSON格式返回，格式为：classifications数组，每个元素包含index和stance字段，stance值为supporting、refuting或neutral。"
    else:
        system = "You are a stance classification expert. Determine whether each piece of evidence supports, refutes, or is neutral toward the given claim.\n\nReturn JSON with a 'classifications' array where each element has 'index' (int) and 'stance' ('supporting', 'refuting', or 'neutral')."

    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Claim: {claim_text}\n\nEvidence:\n{evidence_list}"),
    ])


def _classify_stances(
    claim_text: str, evidence_items: list[Evidence], language: str
) -> dict[int, str]:
    """Classify evidence stances using LLM. Returns {index: stance}."""
    if not evidence_items:
        return {}

    evidence_list = ""
    for i, e in enumerate(evidence_items):
        evidence_list += f"[{i}] {e.content}\n"

    try:
        raw = node_utils.invoke_structured(
            _build_stance_prompt(language),
            {},
            {"claim_text": claim_text, "evidence_list": evidence_list},
        )
        parsed = node_utils.parse_json_response(raw)
        result = {}
        for item in parsed.get("classifications", []):
            idx = item.get("index")
            stance = item.get("stance", "neutral")
            if idx is not None and stance in ("supporting", "refuting", "neutral"):
                result[idx] = stance
        return result
    except Exception:
        # Default: all neutral
        return {i: "neutral" for i in range(len(evidence_items))}


def evidence_aggregator(state: FactCheckState) -> dict:
    """
    Aggregates evidence, deduplicates, classifies stance, detects conflicts.

    State reads: claims, evidence, credibility_scores
    State writes: evidence (replaced with classified, deduplicated evidence)
    """
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])
    credibility_scores = state.get("credibility_scores", [])
    language = state.get("language", "zh")

    if not evidence:
        return {"evidence": []}

    node_utils.emit_progress("evidence_aggregation", f"Aggregating {len(evidence)} evidence items...")

    # Build credibility lookup
    cred_lookup: dict[str, float] = {}
    for cs in credibility_scores:
        url = cs.get("source_url", "")
        if url:
            cred_lookup[url] = cs.get("score", 0.5)

    # Deduplicate by content hash
    seen_hashes: set[str] = set()
    deduplicated: list[Evidence] = []
    for ev in evidence:
        h = _content_hash(ev.content, ev.source_url)
        if h not in seen_hashes:
            seen_hashes.add(h)
            # Update credibility score from scorer
            score = cred_lookup.get(ev.source_url, ev.credibility_score)
            deduplicated.append(
                Evidence(
                    claim_id=ev.claim_id,
                    content=ev.content,
                    source_url=ev.source_url,
                    source_name=ev.source_name,
                    source_type=ev.source_type,
                    stance=ev.stance,
                    credibility_score=score,
                )
            )

    # Classify stances per claim
    for claim in claims:
        claim_evidence = [e for e in deduplicated if e.claim_id == claim.id]
        if not claim_evidence:
            continue

        stance_map = _classify_stances(claim.text, claim_evidence, language)

        for i, ev in enumerate(claim_evidence):
            if i in stance_map:
                idx = deduplicated.index(ev)
                deduplicated[idx] = Evidence(
                    claim_id=ev.claim_id,
                    content=ev.content,
                    source_url=ev.source_url,
                    source_name=ev.source_name,
                    source_type=ev.source_type,
                    stance=stance_map[i],
                    credibility_score=ev.credibility_score,
                )

    # Replace evidence with classified version
    node_utils.emit_progress(
        "evidence_aggregation",
        f"Aggregated to {len(deduplicated)} items (removed {len(evidence) - len(deduplicated)} duplicates)",
    )

    return {"evidence": deduplicated}
