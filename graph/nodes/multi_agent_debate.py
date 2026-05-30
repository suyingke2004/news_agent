from typing import Any, cast

from langchain_core.prompts import ChatPromptTemplate

from graph.nodes.utils import emit_progress, invoke_structured, parse_json_response
from graph.state import DebateArgument, Evidence, FactCheckState


def _format_evidence(
    evidence: list[Evidence], claim_id: str, stance_filter: str | None = None
) -> str:
    lines: list[str] = []
    filtered = [item for item in evidence if item.claim_id == claim_id]
    if stance_filter:
        filtered = [item for item in filtered if item.stance == stance_filter]

    if not filtered:
        return "No evidence available."

    for item in filtered:
        credibility = (
            f" (credibility: {item.credibility_score:.1f})"
            if item.credibility_score != 0.5
            else ""
        )
        lines.append(f"- [{item.source_name}]{credibility} {item.content}")
    return "\n".join(lines)


def _build_advocate_prompt(language: str) -> ChatPromptTemplate:
    if language == "zh":
        system = """你是一个支持方分析师（Advocate）。你的任务是找出支持给定声明为真的证据和论点。

要求：
1. 基于提供的证据，构建支持声明的论点
2. 指出支持证据中最有力的几点
3. 如果证据不足，也要诚实说明
4. 回复格式：论点总结 + 3-5个关键要点"""
    else:
        system = """You are an Advocate analyst. Your role is to find evidence and arguments supporting the claim as TRUE.

Requirements:
1. Based on the provided evidence, construct arguments supporting the claim
2. Highlight the strongest supporting points
3. If evidence is insufficient, acknowledge it honestly
4. Format: argument summary + 3-5 key points"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Claim: {claim_text}\n\nAvailable evidence:\n{evidence}"),
        ]
    )


def _build_skeptic_prompt(language: str) -> ChatPromptTemplate:
    if language == "zh":
        system = """你是一个质疑方分析师（Skeptic）。你的任务是找出反驳给定声明的证据和论点。

要求：
1. 基于提供的证据，构建质疑或反驳声明的论点
2. 指出反驳证据中最有力的几点
3. 分析证据中可能的逻辑漏洞或矛盾
4. 回复格式：论点总结 + 3-5个关键要点"""
    else:
        system = """You are a Skeptic analyst. Your role is to find evidence and arguments refuting the claim.

Requirements:
1. Based on the provided evidence, construct arguments challenging or refuting the claim
2. Highlight the strongest refuting points
3. Analyze potential logical gaps or contradictions in the evidence
4. Format: argument summary + 3-5 key points"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Claim: {claim_text}\n\nAvailable evidence:\n{evidence}"),
        ]
    )


def _build_neutral_prompt(language: str) -> ChatPromptTemplate:
    if language == "zh":
        system = """你是一个中立分析师（Neutral Analyst）。你的任务是客观分析声明和证据，不偏向支持或反驳。

要求：
1. 客观总结所有证据
2. 指出证据中的不确定性和矛盾
3. 评估证据的整体质量和充分程度
4. 回复格式：分析总结 + 3-5个关键要点"""
    else:
        system = """You are a Neutral Analyst. Your role is to objectively analyze the claim and evidence without bias.

Requirements:
1. Objectively summarize all evidence
2. Point out uncertainties and contradictions in the evidence
3. Assess the overall quality and sufficiency of evidence
4. Format: analysis summary + 3-5 key points"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Claim: {claim_text}\n\nAvailable evidence:\n{evidence}"),
        ]
    )


def _build_judge_prompt(language: str) -> ChatPromptTemplate:
    if language == "zh":
        system = "你是一个裁判（Judge）。基于支持方、质疑方和中立分析师的论点，做出最终裁决。\n\n请评估各方论点的强度，并给出一个分歧度分数（0.0-1.0）：\n- 0.0: 各方完全一致\n- 0.5: 存在显著分歧\n- 1.0: 各方观点完全对立\n\n请以JSON格式返回，包含disagreement_score和reasoning字段。"
    else:
        system = "You are a Judge. Based on the Advocate, Skeptic, and Neutral analyst arguments, make a ruling.\n\nEvaluate the strength of each side's arguments and provide a disagreement score (0.0-1.0):\n- 0.0: All sides agree completely\n- 0.5: Significant disagreement exists\n- 1.0: Sides are completely opposed\n\nReturn JSON with 'disagreement_score' and 'reasoning' fields."

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                (
                    "Claim: {claim_text}\n\nAdvocate's argument:\n{advocate_arg}"
                    "\n\nSkeptic's argument:\n{skeptic_arg}"
                    "\n\nNeutral analysis:\n{neutral_arg}"
                ),
            ),
        ]
    )


def _run_debate_for_claim(
    claim_id: str,
    claim_text: str,
    evidence: list[Evidence],
    language: str,
) -> tuple[list[DebateArgument], float]:
    all_evidence_str = _format_evidence(evidence, claim_id)
    supporting_str = _format_evidence(evidence, claim_id, "supporting")
    refuting_str = _format_evidence(evidence, claim_id, "refuting")

    advocate_arg = invoke_structured(
        _build_advocate_prompt(language),
        {},
        {
            "claim_text": claim_text,
            "evidence": f"Supporting:\n{supporting_str}\n\nAll:\n{all_evidence_str}",
        },
    )

    skeptic_arg = invoke_structured(
        _build_skeptic_prompt(language),
        {},
        {
            "claim_text": claim_text,
            "evidence": f"Refuting:\n{refuting_str}\n\nAll:\n{all_evidence_str}",
        },
    )

    neutral_arg = invoke_structured(
        _build_neutral_prompt(language),
        {},
        {"claim_text": claim_text, "evidence": all_evidence_str},
    )

    arguments = [
        DebateArgument(role="advocate", claim_id=claim_id, argument=advocate_arg),
        DebateArgument(role="skeptic", claim_id=claim_id, argument=skeptic_arg),
        DebateArgument(role="neutral", claim_id=claim_id, argument=neutral_arg),
    ]

    try:
        judge_raw = invoke_structured(
            _build_judge_prompt(language),
            {},
            {
                "claim_text": claim_text,
                "advocate_arg": advocate_arg[:500],
                "skeptic_arg": skeptic_arg[:500],
                "neutral_arg": neutral_arg[:500],
            },
        )
        judge_result = parse_json_response(judge_raw)
        if isinstance(judge_result, dict):
            disagreement = float(judge_result.get("disagreement_score", 0.5))
        else:
            disagreement = 0.5
    except Exception:
        disagreement = 0.5

    return arguments, disagreement


def multi_agent_debate(state: FactCheckState) -> dict[str, Any]:
    claims = cast(list[Any], state.get("claims", []))
    evidence = cast(list[Evidence], state.get("evidence", []))
    language = cast(str, state.get("language", "zh"))

    if not claims:
        return {"debate_arguments": [], "disagreement_scores": {}}

    emit_progress("multi_agent_debate", f"Running debate for {len(claims)} claims...")

    all_arguments: list[DebateArgument] = []
    disagreement_scores: dict[str, float] = {}

    for claim in claims:
        try:
            arguments, score = _run_debate_for_claim(
                claim.id, claim.text, evidence, language
            )
            all_arguments.extend(arguments)
            disagreement_scores[claim.id] = score
        except Exception as exc:
            all_arguments.append(
                DebateArgument(
                    role="neutral",
                    claim_id=claim.id,
                    argument=f"Debate error: {exc}",
                )
            )
            disagreement_scores[claim.id] = 0.5

    average_disagreement = sum(disagreement_scores.values()) / max(
        len(disagreement_scores), 1
    )
    emit_progress(
        "multi_agent_debate",
        f"Debate complete. Avg disagreement: {average_disagreement:.2f}",
    )

    return {
        "debate_arguments": all_arguments,
        "disagreement_scores": disagreement_scores,
    }
