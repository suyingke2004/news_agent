import pytest
from unittest.mock import patch, MagicMock
from graph.nodes.evidence_aggregator import (
    evidence_aggregator,
    _content_hash,
    _classify_stances,
)
from graph.state import Claim, Evidence


class TestContentHash:
    def test_same_content_same_hash(self):
        h1 = _content_hash("Apple revenue grew", "https://a.com")
        h2 = _content_hash("Apple revenue grew", "https://a.com")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = _content_hash("Apple revenue grew", "https://a.com")
        h2 = _content_hash("Google revenue grew", "https://a.com")
        assert h1 != h2

    def test_case_insensitive(self):
        h1 = _content_hash("Apple Revenue", "https://a.com")
        h2 = _content_hash("apple revenue", "https://a.com")
        assert h1 == h2


class TestClassifyStances:
    @patch("graph.nodes.utils.invoke_structured")
    def test_classifies_correctly(self, mock_invoke):
        mock_invoke.return_value = '{"classifications": [{"index": 0, "stance": "supporting"}, {"index": 1, "stance": "refuting"}]}'
        
        evidence = [
            Evidence(claim_id="c0", content="Revenue grew 50%"),
            Evidence(claim_id="c0", content="Revenue actually declined"),
        ]
        result = _classify_stances("Revenue grew", evidence, "en")
        assert result[0] == "supporting"
        assert result[1] == "refuting"

    @patch("graph.nodes.utils.invoke_structured")
    def test_exception_returns_neutral(self, mock_invoke):
        mock_invoke.side_effect = Exception("LLM error")
        
        evidence = [Evidence(claim_id="c0", content="Something")]
        result = _classify_stances("Claim", evidence, "en")
        assert result[0] == "neutral"

    def test_empty_evidence(self):
        result = _classify_stances("Claim", [], "en")
        assert result == {}


class TestEvidenceAggregator:
    @patch("graph.nodes.evidence_aggregator._classify_stances")
    def test_deduplicates(self, mock_classify):
        mock_classify.return_value = {0: "supporting"}
        
        evidence = [
            Evidence(claim_id="c0", content="Same content", source_url="https://a.com"),
            Evidence(claim_id="c0", content="Same content", source_url="https://a.com"),
            Evidence(claim_id="c0", content="Different content", source_url="https://b.com"),
        ]
        state = {
            "claims": [Claim(id="c0", text="Test")],
            "evidence": evidence,
            "credibility_scores": [],
            "language": "en",
        }
        result = evidence_aggregator(state)
        
        assert len(result["evidence"]) == 2

    def test_empty_evidence(self):
        state = {
            "claims": [],
            "evidence": [],
            "credibility_scores": [],
            "language": "en",
        }
        result = evidence_aggregator(state)
        assert result["evidence"] == []

    @patch("graph.nodes.evidence_aggregator._classify_stances")
    def test_updates_credibility_scores(self, mock_classify):
        mock_classify.return_value = {0: "neutral"}
        
        evidence = [
            Evidence(claim_id="c0", content="Test", source_url="https://reuters.com/a"),
        ]
        cred_scores = [{"source_url": "https://reuters.com/a", "score": 0.9, "source_name": "Reuters"}]
        
        state = {
            "claims": [Claim(id="c0", text="Test")],
            "evidence": evidence,
            "credibility_scores": cred_scores,
            "language": "en",
        }
        result = evidence_aggregator(state)
        
        assert result["evidence"][0].credibility_score == 0.9
