"""Edge case tests for claim_extractor node."""
import importlib
from unittest.mock import MagicMock, patch
from collections.abc import Callable
from typing import cast

from graph.nodes.claim_extractor import claim_extractor
from graph.state import Claim, FactCheckState

claim_extractor_module = importlib.import_module("graph.nodes.claim_extractor")
detect_input_type = cast(Callable[[str], str], getattr(claim_extractor_module, "_detect_input_type"))
fetch_url_content = cast(Callable[[str], str], getattr(claim_extractor_module, "_fetch_url_content"))


class TestDetectInputType:
    def test_url_input(self):
        assert detect_input_type("https://example.com/article") == "url"

    def test_http_url(self):
        assert detect_input_type("http://example.com/article") == "url"

    def test_short_claim(self):
        assert detect_input_type("The Earth is flat") == "claim"

    def test_long_text(self):
        text = "This is a longer piece of text that spans multiple sentences and contains enough words to be classified as text input rather than a simple claim or URL."
        assert detect_input_type(text) == "text"


class TestFetchUrlContent:
    @patch("graph.nodes.claim_extractor.Article")
    def test_fetch_failure_returns_error_string(self, mock_article_cls: MagicMock):
        mock_article = MagicMock()
        mock_article.download.side_effect = Exception("Connection timeout")
        mock_article.parse = MagicMock()
        mock_article_cls.return_value = mock_article
        result = fetch_url_content("https://nonexistent.invalid")
        assert "[Error fetching URL:" in result

    @patch("graph.nodes.claim_extractor.Article")
    def test_fetch_success_returns_text(self, mock_article_cls: MagicMock):
        mock_article = MagicMock()
        mock_article.text = "Article content here"
        mock_article_cls.return_value = mock_article
        result = fetch_url_content("https://example.com")
        assert result == "Article content here"


class TestClaimExtractorEdgeCases:
    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_empty_input_returns_fallback_claim(self, mock_invoke: MagicMock):
        """Empty string input should still produce a claim via LLM fallback."""
        mock_invoke.return_value = '{"claims": []}'
        state = cast(FactCheckState, cast(object, {"raw_input": "", "language": "en"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])
        assert "claims" in result
        assert len(claims) >= 1  # Fallback to single claim

    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_malformed_llm_json_returns_fallback(self, mock_invoke: MagicMock):
        """When LLM returns invalid JSON, fallback to single claim."""
        mock_invoke.return_value = "not json at all"
        state = cast(FactCheckState, cast(object, {"raw_input": "Some claim text", "language": "en"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])
        assert "claims" in result
        assert len(claims) >= 1

    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_llm_exception_returns_fallback(self, mock_invoke: MagicMock):
        """When LLM throws, fallback to single claim with raw_input."""
        mock_invoke.side_effect = Exception("LLM error")
        state = cast(FactCheckState, cast(object, {"raw_input": "Test claim", "language": "en"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])
        assert "claims" in result
        assert claims[0].text == "Test claim"
        assert "errors" in result

    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_chinese_input(self, mock_invoke: MagicMock):
        """Chinese text should work correctly."""
        mock_invoke.return_value = '{"claims": [{"id": "claim_0", "text": "人工智能将取代50%的工作"}]}'
        state = cast(FactCheckState, cast(object, {"raw_input": "据媒体报道，人工智能将在2030年取代50%的工作岗位", "language": "zh"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])
        assert len(claims) == 1
        assert "人工智能" in claims[0].text
