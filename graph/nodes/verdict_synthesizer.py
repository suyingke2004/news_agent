from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from graph.nodes.utils import emit_progress, invoke_structured, parse_json_response
from graph.state import Claim, DebateArgument, Evidence, FactCheckState, Verdict


def _safe_emit_progress(stage: str, message: str) -> None:
    try:
        emit_progress(stage, message)
    except Exception:
        pass


def _build_verdict_prompt(language: str) -> ChatPromptTemplate:
    if language == "zh":
        system = "你是一个事实核查裁判。基于收集到的证据和多角度辩论，对每个声明做出最终裁决。\n\n裁决类型：\n- TRUE: 声明基本属实\n- FALSE: 声明基本不实\n- MIXED: 声明部分属实，部分不实\n- UNVERIFIED: 证据不足以做出明确判断\n\n请以JSON格式返回，包含verdict、confidence(0-1)和reasoning字段。"
    else:
        system = "You are a fact-check judge. Based on collected evidence and multi-perspective debate, make a final verdict for each claim.\n\nVerdict types:\n- TRUE: The claim is mostly accurate with sufficient supporting evidence\n- FALSE: The claim is mostly inaccurate with sufficient refuting evidence\n- MIXED: The claim is partially accurate, partially inaccurate\n- UNVERIFIED: Insufficient evidence to make a definitive judgment\n\nReturn JSON with 'verdict', 'confidence' (0-1), and 'reasoning' fields."

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """Claim: {claim_text}

Evidence summary:
{evidence_summary}

Debate summary:
{debate_summary}

Disagreement score: {disagreement_score}

Make your verdict:""",
            ),
        ]
    )


def _summarize_evidence(evidence: list[Evidence], claim_id: str) -> str:
    claim_evidence = [e for e in evidence if e.claim_id == claim_id]
    if not claim_evidence:
        return "No evidence found."

    supporting = [e for e in claim_evidence if e.stance == "supporting"]
    refuting = [e for e in claim_evidence if e.stance == "refuting"]
    neutral = [e for e in claim_evidence if e.stance == "neutral"]

    lines = [
        f"Total evidence items: {len(claim_evidence)}",
        f"Supporting: {len(supporting)}, Refuting: {len(refuting)}, Neutral: {len(neutral)}",
    ]

    for item in claim_evidence[:10]:
        lines.append(f"- [{item.source_name}] ({item.stance}) {item.content[:100]}")

    return "\n".join(lines)


def _summarize_debate(arguments: list[DebateArgument], claim_id: str) -> str:
    claim_args = [a for a in arguments if a.claim_id == claim_id]
    if not claim_args:
        return "No debate arguments."

    lines = []
    for arg in claim_args:
        lines.append(f"[{arg.role.upper()}] {arg.argument[:200]}")
    return "\n".join(lines)


def _compute_overall_verdict(verdicts: list[Verdict]) -> tuple[str, float]:
    if not verdicts:
        return "UNVERIFIED", 0.0

    verdict_types = {v.verdict for v in verdicts}

    if verdict_types == {"TRUE"}:
        overall = "TRUE"
    elif verdict_types == {"FALSE"}:
        overall = "FALSE"
    elif verdict_types == {"UNVERIFIED"}:
        overall = "UNVERIFIED"
    elif "FALSE" in verdict_types and "TRUE" in verdict_types:
        overall = "MIXED"
    elif "MIXED" in verdict_types:
        overall = "MIXED"
    else:
        overall = "MIXED"

    overall_confidence = sum(v.confidence for v in verdicts) / len(verdicts)
    return overall, round(overall_confidence, 2)


def _generate_report_markdown(
    claims: list[Claim],
    verdicts: list[Verdict],
    evidence: list[Evidence],
    debate_arguments: list[DebateArgument],
    overall_verdict: str,
    overall_confidence: float,
    language: str,
) -> str:
    verdict_emoji = {"TRUE": "✅", "FALSE": "❌", "MIXED": "⚠️", "UNVERIFIED": "❓"}

    if language == "zh":
        lines = [
            "# 🔍 事实核查报告\n",
            f"## 总体结论: {verdict_emoji.get(overall_verdict, '❓')} {overall_verdict}",
            f"**置信度**: {overall_confidence:.0%}\n",
        ]

        for i, verdict in enumerate(verdicts):
            claim = next((c for c in claims if c.id == verdict.claim_id), None)
            claim_text = claim.text if claim else verdict.claim_id

            lines.append(f"### 声明 {i + 1}: {claim_text}")
            lines.append(f"**裁决**: {verdict_emoji.get(verdict.verdict, '❓')} {verdict.verdict}")
            lines.append(f"**置信度**: {verdict.confidence:.0%}")
            lines.append(
                f"**支持证据**: {verdict.supporting_evidence_count} | **反驳证据**: {verdict.refuting_evidence_count}"
            )
            lines.append(f"**推理**: {verdict.reasoning}\n")

            claim_evidence = [e for e in evidence if e.claim_id == verdict.claim_id]
            if claim_evidence:
                lines.append("**证据**:")
                for item in claim_evidence[:5]:
                    lines.append(f"- [{item.source_name}] {item.content[:150]}")
                lines.append("")

            claim_args = [a for a in debate_arguments if a.claim_id == verdict.claim_id]
            if claim_args:
                lines.append("**辩论摘要**:")
                for arg in claim_args:
                    lines.append(f"- **{arg.role.upper()}**: {arg.argument[:150]}")
                lines.append("")
    else:
        lines = [
            "# 🔍 Fact-Check Report\n",
            f"## Overall Verdict: {verdict_emoji.get(overall_verdict, '❓')} {overall_verdict}",
            f"**Confidence**: {overall_confidence:.0%}\n",
        ]

        for i, verdict in enumerate(verdicts):
            claim = next((c for c in claims if c.id == verdict.claim_id), None)
            claim_text = claim.text if claim else verdict.claim_id

            lines.append(f"### Claim {i + 1}: {claim_text}")
            lines.append(f"**Verdict**: {verdict_emoji.get(verdict.verdict, '❓')} {verdict.verdict}")
            lines.append(f"**Confidence**: {verdict.confidence:.0%}")
            lines.append(
                f"**Supporting**: {verdict.supporting_evidence_count} | **Refuting**: {verdict.refuting_evidence_count}"
            )
            lines.append(f"**Reasoning**: {verdict.reasoning}\n")

            claim_evidence = [e for e in evidence if e.claim_id == verdict.claim_id]
            if claim_evidence:
                lines.append("**Evidence**:")
                for item in claim_evidence[:5]:
                    lines.append(f"- [{item.source_name}] {item.content[:150]}")
                lines.append("")

            claim_args = [a for a in debate_arguments if a.claim_id == verdict.claim_id]
            if claim_args:
                lines.append("**Debate Summary**:")
                for arg in claim_args:
                    lines.append(f"- **{arg.role.upper()}**: {arg.argument[:150]}")
                lines.append("")

    return "\n".join(lines)


def verdict_synthesizer(state: FactCheckState) -> dict[str, Any]:
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])
    debate_arguments = state.get("debate_arguments", [])
    disagreement_scores = state.get("disagreement_scores", {})
    language = state.get("language", "zh")

    if not claims:
        return {
            "verdicts": [],
            "overall_verdict": "UNVERIFIED",
            "overall_confidence": 0.0,
            "report_markdown": "# No claims to verify",
        }

    _safe_emit_progress(
        "verdict_synthesis", f"Generating verdicts for {len(claims)} claims..."
    )

    verdicts: list[Verdict] = []
    for claim in claims:
        evidence_summary = _summarize_evidence(evidence, claim.id)
        debate_summary = _summarize_debate(debate_arguments, claim.id)
        disagreement = disagreement_scores.get(claim.id, 0.5)

        try:
            raw = invoke_structured(
                _build_verdict_prompt(language),
                {},
                {
                    "claim_text": claim.text,
                    "evidence_summary": evidence_summary,
                    "debate_summary": debate_summary,
                    "disagreement_score": str(disagreement),
                },
            )
            parsed = parse_json_response(raw)

            verdict = Verdict(
                claim_id=claim.id,
                verdict=parsed.get("verdict", "UNVERIFIED"),
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
                supporting_evidence_count=len(
                    [
                        e
                        for e in evidence
                        if e.claim_id == claim.id and e.stance == "supporting"
                    ]
                ),
                refuting_evidence_count=len(
                    [e for e in evidence if e.claim_id == claim.id and e.stance == "refuting"]
                ),
            )
        except Exception as exc:
            verdict = Verdict(
                claim_id=claim.id,
                verdict="UNVERIFIED",
                confidence=0.0,
                reasoning=f"Error generating verdict: {exc}",
            )

        verdicts.append(verdict)

    overall_verdict, overall_confidence = _compute_overall_verdict(verdicts)
    report = _generate_report_markdown(
        claims,
        verdicts,
        evidence,
        debate_arguments,
        overall_verdict,
        overall_confidence,
        language,
    )

    _safe_emit_progress(
        "verdict_synthesis",
        f"Verdict: {overall_verdict} (confidence: {overall_confidence:.0%})",
    )

    return {
        "verdicts": verdicts,
        "overall_verdict": overall_verdict,
        "overall_confidence": overall_confidence,
        "report_markdown": report,
    }
