import pytest
from unittest.mock import patch
from graph.nodes.source_credibility import (
    source_credibility,
    _extract_domain,
    _score_domain,
    TIER_1_DOMAINS,
    TIER_2_DOMAINS,
    TIER_3_DOMAINS,
)
from graph.state import Evidence


class TestExtractDomain:
    def test_standard_url(self):
        assert _extract_domain("https://www.reuters.com/article/123") == "reuters.com"

    def test_no_www(self):
        assert _extract_domain("https://bbc.com/news/world") == "bbc.com"

    def test_empty_url(self):
        assert _extract_domain("") == ""

    def test_malformed_url(self):
        result = _extract_domain("not-a-url")
        assert isinstance(result, str)


class TestScoreDomain:
    def test_tier_1_reuters(self):
        score, reasons = _score_domain("reuters.com")
        assert score == 0.9
        assert any("tier_1" in r for r in reasons)

    def test_tier_1_bbc(self):
        score, reasons = _score_domain("www.bbc.com")
        assert score == 0.9

    def test_tier_2_cnn(self):
        score, reasons = _score_domain("www.cnn.com")
        assert score == 0.75
        assert any("tier_2" in r for r in reasons)

    def test_tier_3_reddit(self):
        score, reasons = _score_domain("www.reddit.com")
        assert score == 0.4
        assert any("tier_3" in r for r in reasons)

    def test_unknown_domain(self):
        score, reasons = _score_domain("random-blog.com")
        assert score == 0.5
        assert "unknown_domain" in reasons

    def test_empty_domain(self):
        score, reasons = _score_domain("")
        assert score == 0.3


class TestSourceCredibility:
    def test_empty_evidence(self):
        state = {"evidence": []}
        with patch("graph.nodes.source_credibility.emit_progress"):
            result = source_credibility(state)
        assert result["credibility_scores"] == []

    def test_scores_evidence_sources(self):
        evidence = [
            Evidence(claim_id="c0", content="test", source_url="https://www.reuters.com/article/1", source_name="Reuters"),
            Evidence(claim_id="c0", content="test2", source_url="https://www.reddit.com/r/news", source_name="Reddit"),
        ]
        state = {"evidence": evidence}
        with patch("graph.nodes.source_credibility.emit_progress"):
            result = source_credibility(state)
        
        assert len(result["credibility_scores"]) == 2
        # Find Reuters score
        reuters = next((s for s in result["credibility_scores"] if "Reuters" in s["source_name"]), None)
        assert reuters is not None
        assert reuters["score"] == 0.9
        
        # Find Reddit score
        reddit = next((s for s in result["credibility_scores"] if "Reddit" in s["source_name"]), None)
        assert reddit is not None
        assert reddit["score"] == 0.4

    def test_deduplicates_same_domain(self):
        evidence = [
            Evidence(claim_id="c0", content="a", source_url="https://www.reuters.com/a", source_name="Reuters"),
            Evidence(claim_id="c0", content="b", source_url="https://www.reuters.com/b", source_name="Reuters"),
        ]
        state = {"evidence": evidence}
        with patch("graph.nodes.source_credibility.emit_progress"):
            result = source_credibility(state)
        
        # Same domain should only appear once
        assert len(result["credibility_scores"]) == 1
