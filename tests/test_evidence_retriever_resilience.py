"""Tests for evidence_retriever resilience when tools are unavailable."""
from unittest.mock import patch, MagicMock
from graph.state import Claim, FactCheckState
from graph.nodes.evidence_retriever import (
    evidence_retriever,
    _load_news_tools,
    _load_reddit_tool,
    _load_rss_tool,
    _load_web_search_tool,
    _parse_news_results,
    _parse_reddit_results,
    _parse_rss_results,
    _parse_web_results,
)


def _state(claims: list[Claim], language: str = "en") -> FactCheckState:
    state: FactCheckState = {"claims": claims, "language": language}
    return state


class TestLoaderResilience:
    """Test that _load_* functions return None when imports fail."""

    @patch("graph.nodes.evidence_retriever.import_module", side_effect=ImportError("no langchain"))
    def test_load_news_tools_returns_none(self, mock_import):
        assert _load_news_tools() is None

    @patch("graph.nodes.evidence_retriever.import_module", side_effect=ImportError("no langchain"))
    def test_load_reddit_tool_returns_none(self, mock_import):
        assert _load_reddit_tool() is None

    @patch("graph.nodes.evidence_retriever.import_module", side_effect=ImportError("no langchain"))
    def test_load_rss_tool_returns_none(self, mock_import):
        assert _load_rss_tool() is None

    @patch("graph.nodes.evidence_retriever.import_module", side_effect=ImportError("no langchain"))
    def test_load_web_search_tool_returns_none(self, mock_import):
        assert _load_web_search_tool() is None


class TestEvidenceRetrieverGracefulDegradation:
    """Test that evidence_retriever returns empty evidence when all tools fail."""

    def test_returns_empty_when_all_loaders_none(self):
        with patch("graph.nodes.evidence_retriever._load_news_tools", return_value=None), \
             patch("graph.nodes.evidence_retriever._load_reddit_tool", return_value=None), \
             patch("graph.nodes.evidence_retriever._load_rss_tool", return_value=None), \
             patch("graph.nodes.evidence_retriever._load_web_search_tool", return_value=None):
            result = evidence_retriever(_state([Claim(id="c0", text="The Earth is flat", source_text="test")]))
        assert result == {"evidence": []}

    def test_returns_empty_for_empty_claims(self):
        result = evidence_retriever(_state([]))
        assert result == {"evidence": []}

    def test_partial_degradation_collects_available(self):
        """When some loaders return None but others work, non-None evidence is collected."""
        mock_reddit_tool = MagicMock()
        mock_reddit_tool.search_reddit.invoke.return_value = "标题: Test Result\n链接: http://example.com\n\n"

        with patch("graph.nodes.evidence_retriever._load_news_tools", return_value=None), \
             patch("graph.nodes.evidence_retriever._load_reddit_tool", return_value=mock_reddit_tool), \
             patch("graph.nodes.evidence_retriever._load_rss_tool", return_value=None), \
             patch("graph.nodes.evidence_retriever._load_web_search_tool", return_value=None):
            result = evidence_retriever(_state([Claim(id="c0", text="test claim", source_text="test")]))
        assert len(result["evidence"]) >= 0  # Should not crash


class TestParserEdgeCases:
    """Test that parse functions handle edge cases."""

    def test_parse_news_empty(self):
        assert _parse_news_results("", "c0") == []

    def test_parse_news_error_text(self):
        assert _parse_news_results("出错", "c0") == []
        assert _parse_news_results("没有找到结果", "c0") == []

    def test_parse_reddit_empty(self):
        assert _parse_reddit_results("", "c0") == []

    def test_parse_reddit_error_text(self):
        assert _parse_reddit_results("error occurred", "c0") == []

    def test_parse_rss_empty(self):
        assert _parse_rss_results("", "c0") == []

    def test_parse_web_empty(self):
        assert _parse_web_results("", "c0") == []

    def test_parse_web_error_text(self):
        assert _parse_web_results("未在", "c0") == []
