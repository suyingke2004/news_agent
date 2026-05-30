from unittest.mock import patch

from graph.nodes.verdict_synthesizer import (
    _compute_overall_verdict,
    _generate_report_markdown,
    _summarize_evidence,
    verdict_synthesizer,
)
from graph.state import Claim, DebateArgument, Evidence, Verdict


class TestComputeOverallVerdict:
    def test_all_true(self):
        verdicts = [
            Verdict(claim_id="c0", verdict="TRUE", confidence=0.9, reasoning="ok"),
            Verdict(claim_id="c1", verdict="TRUE", confidence=0.8, reasoning="ok"),
        ]
        overall, conf = _compute_overall_verdict(verdicts)
        assert overall == "TRUE"
        assert conf == 0.85

    def test_all_false(self):
        verdicts = [
            Verdict(claim_id="c0", verdict="FALSE", confidence=0.9, reasoning="ok"),
        ]
        overall, conf = _compute_overall_verdict(verdicts)
        assert overall == "FALSE"

    def test_mixed_true_and_false(self):
        verdicts = [
            Verdict(claim_id="c0", verdict="TRUE", confidence=0.8, reasoning="ok"),
            Verdict(claim_id="c1", verdict="FALSE", confidence=0.7, reasoning="ok"),
        ]
        overall, conf = _compute_overall_verdict(verdicts)
        assert overall == "MIXED"

    def test_empty_verdicts(self):
        overall, conf = _compute_overall_verdict([])
        assert overall == "UNVERIFIED"
        assert conf == 0.0


class TestSummarizeEvidence:
    def test_summarizes_evidence(self):
        evidence = [
            Evidence(
                claim_id="c0",
                content="Supporting fact",
                source_name="Reuters",
                stance="supporting",
            ),
            Evidence(
                claim_id="c0",
                content="Refuting fact",
                source_name="Blog",
                stance="refuting",
            ),
        ]
        result = _summarize_evidence(evidence, "c0")
        assert "Total evidence items: 2" in result
        assert "Supporting: 1" in result

    def test_no_evidence(self):
        result = _summarize_evidence([], "c0")
        assert "No evidence found" in result


class TestVerdictSynthesizer:
    @patch("graph.nodes.verdict_synthesizer.invoke_structured")
    def test_generates_verdict_per_claim(self, mock_invoke):
        mock_invoke.return_value = (
            '{"verdict": "TRUE", "confidence": 0.85, "reasoning": '
            '"Strong evidence supports the claim"}'
        )

        state = {
            "claims": [Claim(id="c0", text="The earth is round")],
            "evidence": [
                Evidence(
                    claim_id="c0",
                    content="Scientific consensus",
                    stance="supporting",
                )
            ],
            "debate_arguments": [
                DebateArgument(
                    role="advocate",
                    claim_id="c0",
                    argument="Strong support",
                )
            ],
            "disagreement_scores": {"c0": 0.2},
            "language": "en",
        }
        result = verdict_synthesizer(state)

        assert len(result["verdicts"]) == 1
        assert result["verdicts"][0].verdict == "TRUE"
        assert result["overall_verdict"] == "TRUE"
        assert 0.0 <= result["overall_confidence"] <= 1.0
        assert len(result["report_markdown"]) > 100

    def test_empty_claims(self):
        state = {
            "claims": [],
            "evidence": [],
            "debate_arguments": [],
            "disagreement_scores": {},
            "language": "en",
        }
        result = verdict_synthesizer(state)

        assert result["overall_verdict"] == "UNVERIFIED"
        assert result["verdicts"] == []

    @patch("graph.nodes.verdict_synthesizer.invoke_structured")
    def test_handles_llm_error(self, mock_invoke):
        mock_invoke.side_effect = Exception("LLM unavailable")

        state = {
            "claims": [Claim(id="c0", text="Test claim")],
            "evidence": [],
            "debate_arguments": [],
            "disagreement_scores": {},
            "language": "en",
        }
        result = verdict_synthesizer(state)

        assert len(result["verdicts"]) == 1
        assert result["verdicts"][0].verdict == "UNVERIFIED"
        assert result["overall_verdict"] == "UNVERIFIED"

    def test_generates_chinese_report(self):
        verdicts = [
            Verdict(claim_id="c0", verdict="TRUE", confidence=0.9, reasoning="充分证据支持")
        ]
        report = _generate_report_markdown(
            [Claim(id="c0", text="地球是圆的")],
            verdicts,
            [],
            [],
            "TRUE",
            0.9,
            "zh",
        )
        assert "事实核查报告" in report
        assert "地球是圆的" in report
