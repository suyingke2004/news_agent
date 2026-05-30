from unittest.mock import patch

from graph.nodes.multi_agent_debate import (
    _format_evidence,
    _run_debate_for_claim,
    multi_agent_debate,
)
from graph.state import Claim, DebateArgument, Evidence


class TestFormatEvidence:
    def test_formats_evidence(self):
        evidence = [
            Evidence(
                claim_id="c0",
                content="Apple revenue grew 50%",
                source_name="Reuters",
                credibility_score=0.9,
            ),
            Evidence(
                claim_id="c0",
                content="Disputed growth figure",
                source_name="Blog",
                stance="refuting",
            ),
        ]
        result = _format_evidence(evidence, "c0")
        assert "Apple revenue grew 50%" in result
        assert "Reuters" in result

    def test_filters_by_stance(self):
        evidence = [
            Evidence(claim_id="c0", content="Supporting", stance="supporting"),
            Evidence(claim_id="c0", content="Refuting", stance="refuting"),
        ]
        result = _format_evidence(evidence, "c0", "supporting")
        assert "Supporting" in result
        assert "Refuting" not in result

    def test_no_evidence(self):
        result = _format_evidence([], "c0")
        assert result == "No evidence available."


class TestRunDebateForClaim:
    @patch("graph.nodes.multi_agent_debate.invoke_structured")
    def test_returns_three_arguments(self, mock_invoke):
        mock_invoke.side_effect = [
            "Advocate says true",
            "Skeptic says false",
            "Neutral says uncertain",
            '{"disagreement_score": 0.7, "reasoning": "Significant disagreement"}',
        ]

        arguments, score = _run_debate_for_claim("c0", "Test claim", [], "en")

        assert len(arguments) == 3
        assert arguments[0].role == "advocate"
        assert arguments[1].role == "skeptic"
        assert arguments[2].role == "neutral"
        assert score == 0.7

    @patch("graph.nodes.multi_agent_debate.invoke_structured")
    def test_judge_failure_defaults_to_0_5(self, mock_invoke):
        mock_invoke.side_effect = [
            "Advocate arg",
            "Skeptic arg",
            "Neutral arg",
            Exception("Parse error"),
        ]

        _, score = _run_debate_for_claim("c0", "Test", [], "en")
        assert score == 0.5


class TestMultiAgentDebate:
    @patch("graph.nodes.multi_agent_debate.emit_progress")
    @patch("graph.nodes.multi_agent_debate._run_debate_for_claim")
    def test_runs_debate_for_each_claim(self, mock_debate, _mock_progress):
        mock_debate.return_value = (
            [
                DebateArgument(role="advocate", claim_id="c0", argument="Pro"),
                DebateArgument(role="skeptic", claim_id="c0", argument="Con"),
                DebateArgument(role="neutral", claim_id="c0", argument="Mid"),
            ],
            0.6,
        )

        state = {
            "claims": [Claim(id="c0", text="Test claim")],
            "evidence": [Evidence(claim_id="c0", content="Some evidence")],
            "language": "en",
        }
        result = multi_agent_debate(state)

        assert len(result["debate_arguments"]) == 3
        assert result["disagreement_scores"]["c0"] == 0.6

    def test_empty_claims(self):
        state = {"claims": [], "evidence": [], "language": "en"}
        result = multi_agent_debate(state)
        assert result["debate_arguments"] == []
        assert result["disagreement_scores"] == {}
