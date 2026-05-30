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
    def test_url_http(self):
        assert detect_input_type("http://example.com/article") == "url"

    def test_url_https(self):
        assert detect_input_type("https://example.com/article") == "url"

    def test_long_text(self):
        assert detect_input_type("x" * 300) == "text"

    def test_short_claim(self):
        assert detect_input_type("The earth is flat") == "claim"

    def test_whitespace_handling(self):
        assert detect_input_type("  https://example.com  ") == "url"


class TestFetchUrlContent:
    @patch("graph.nodes.claim_extractor.Article")
    def test_success(self, mock_article_cls: MagicMock):
        mock_article = MagicMock()
        mock_article.text = "Article content here"
        mock_article_cls.return_value = mock_article

        result = fetch_url_content("https://example.com")
        assert result == "Article content here"

    @patch("graph.nodes.claim_extractor.Article")
    def test_failure_returns_error(self, mock_article_cls: MagicMock):
        mock_article_cls.side_effect = Exception("Network error")
        result = fetch_url_content("https://example.com")
        assert "Error" in result


class TestClaimExtractor:
    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_extracts_multiple_claims(self, mock_invoke: MagicMock):
        mock_invoke.return_value = (
            '{"claims": [{"id": "claim_0", "text": "Revenue grew 50%"}, '
            '{"id": "claim_1", "text": "CEO resigned"}]}'
        )

        state = cast(FactCheckState, cast(object, {"raw_input": "Some article text about a company", "language": "en"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])

        assert len(claims) == 2
        assert claims[0].id == "claim_0"
        assert claims[0].text == "Revenue grew 50%"
        assert result["input_type"] == "text"

    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_fallback_on_empty_response(self, mock_invoke: MagicMock):
        mock_invoke.return_value = '{"claims": []}'

        state = cast(FactCheckState, cast(object, {"raw_input": "A claim", "language": "en"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])

        assert len(claims) == 1
        assert claims[0].text == "A claim"

    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_fallback_on_exception(self, mock_invoke: MagicMock):
        mock_invoke.side_effect = Exception("LLM error")

        state = cast(FactCheckState, cast(object, {"raw_input": "A claim", "language": "en"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])

        assert len(claims) == 1
        assert "errors" in result

    @patch("graph.nodes.claim_extractor.invoke_structured")
    @patch("graph.nodes.claim_extractor._fetch_url_content")
    def test_url_input_fetches_content(self, mock_fetch: MagicMock, mock_invoke: MagicMock):
        mock_fetch.return_value = "Fetched article content"
        mock_invoke.return_value = (
            '{"claims": [{"id": "claim_0", "text": "Some claim from article"}]}'
        )

        state = cast(FactCheckState, cast(object, {"raw_input": "https://example.com/news", "language": "en"}))
        result = claim_extractor(state)

        assert result["input_type"] == "url"
        mock_fetch.assert_called_once_with("https://example.com/news")

    @patch("graph.nodes.claim_extractor.invoke_structured")
    def test_chinese_prompt(self, mock_invoke: MagicMock):
        mock_invoke.return_value = '{"claims": [{"id": "claim_0", "text": "公司收入增长50%"}]}'

        state = cast(FactCheckState, cast(object, {"raw_input": "某公司收入大幅增长", "language": "zh"}))
        result = claim_extractor(state)
        claims = cast(list[Claim], result["claims"])

        assert len(claims) == 1
        assert claims[0].text == "公司收入增长50%"
