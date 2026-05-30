from typing import Protocol, cast
from unittest.mock import MagicMock, patch

import graph.nodes.evidence_retriever as evidence_retriever_module
from graph.state import Claim, Evidence, FactCheckState


class _Invokable(Protocol):
    def invoke(self, payload: dict[str, str]) -> str: ...


class _NewsParseFn(Protocol):
    def __call__(self, raw_text: str, claim_id: str) -> list[Evidence]: ...


class _RssParseFn(Protocol):
    def __call__(
        self, raw_text: str, claim_id: str, source_name: str = "RSS Feed"
    ) -> list[Evidence]: ...


evidence_retriever = evidence_retriever_module.evidence_retriever
parse_news_results = cast(
    _NewsParseFn,
    object.__getattribute__(evidence_retriever_module, "_parse_news_results"),
)
parse_reddit_results = cast(
    _NewsParseFn,
    object.__getattribute__(evidence_retriever_module, "_parse_reddit_results"),
)
parse_rss_results = cast(
    _RssParseFn,
    object.__getattribute__(evidence_retriever_module, "_parse_rss_results"),
)
parse_web_results = cast(
    _NewsParseFn,
    object.__getattribute__(evidence_retriever_module, "_parse_web_results"),
)


class TestParseNewsResults:
    def test_parses_multiple_entries(self):
        raw = "标题: Apple revenue grows\n链接: https://example.com/1\n\n标题: iPhone sales surge\n链接: https://example.com/2"
        results = parse_news_results(raw, "c0")
        assert len(results) == 2
        assert results[0].content == "Apple revenue grows"
        assert results[0].source_url == "https://example.com/1"
        assert results[0].source_name == "NewsAPI"
        assert results[0].claim_id == "c0"

    def test_empty_input(self):
        assert parse_news_results("", "c0") == []

    def test_no_results_message(self):
        assert parse_news_results("没有找到相关的新闻。", "c0") == []


class TestParseRedditResults:
    def test_parses_results(self):
        raw = "标题: Discussion about Apple\n链接: https://reddit.com/1\n评分: 12\n\n标题: Another thread\n链接: https://reddit.com/2\n评分: 4"
        results = parse_reddit_results(raw, "c0")
        assert len(results) == 2
        assert results[0].source_type == "reddit"
        assert results[0].source_url == "https://reddit.com/1"

    def test_no_results(self):
        assert parse_reddit_results("没有找到", "c0") == []


class TestParseRssResults:
    def test_parses_results(self):
        raw = "RSS订阅源: BBC\n链接: https://bbc.com/rss\n\n标题: Breaking update\n链接: https://bbc.com/1\n发布日期: 2024-01-01"
        results = parse_rss_results(raw, "c0", source_name="BBC")
        assert len(results) == 1
        assert results[0].source_name == "BBC"
        assert results[0].source_type == "rss"

    def test_no_results(self):
        assert parse_rss_results("RSS订阅源中没有找到条目。", "c0") == []


class TestParseWebResults:
    def test_parses_results(self):
        raw = "来源: BBC\n标题: Breaking news\n链接: https://bbc.com/1"
        results = parse_web_results(raw, "c0")
        assert len(results) == 1
        assert results[0].source_name == "BBC"
        assert results[0].source_type == "web"

    def test_no_results(self):
        assert parse_web_results("未在常见新闻网站上找到相关结果。", "c0") == []


def _state(claims: list[Claim], language: str = "en") -> FactCheckState:
    state: FactCheckState = {"claims": claims, "language": language}
    return state


class TestEvidenceRetriever:
    def test_empty_claims(self):
        result = evidence_retriever(_state([]))
        assert result["evidence"] == []

    @patch("graph.nodes.evidence_retriever._load_web_search_tool")
    @patch("graph.nodes.evidence_retriever._load_rss_tool")
    @patch("graph.nodes.evidence_retriever._load_reddit_tool")
    @patch("graph.nodes.evidence_retriever._load_news_tools")
    def test_searches_all_sources(
        self,
        mock_news_tools: MagicMock,
        mock_reddit_tool: MagicMock,
        mock_rss_tool: MagicMock,
        mock_web_tool: MagicMock,
    ):
        news_tools = MagicMock()
        reddit_tool = MagicMock()
        rss_tool = MagicMock()
        web_tool = MagicMock()
        mock_news_tools.return_value = news_tools
        mock_reddit_tool.return_value = reddit_tool
        mock_rss_tool.return_value = rss_tool
        mock_web_tool.return_value = web_tool

        news_tools.search_news.invoke.return_value = (
            "标题: Apple revenue grows\n链接: https://example.com/1"
        )
        reddit_tool.search_reddit.invoke.return_value = (
            "标题: Apple thread\n链接: https://reddit.com/1\n评分: 88"
        )
        rss_tool.search_rss_feeds.invoke.return_value = (
            "RSS订阅源: BBC\n链接: https://bbc.com/rss\n\n"
            "标题: Apple grew 50% in latest report\n链接: https://bbc.com/apple"
        )
        web_tool.invoke.return_value = (
            "来源: BBC\n标题: Apple revenue grows\n链接: https://bbc.com/1"
        )

        result = evidence_retriever(_state([Claim(id="c0", text="Apple grew 50%")]))

        news_tools.search_news.invoke.assert_called_once_with(
            {"query": "Apple grew 50%"}
        )
        reddit_tool.search_reddit.invoke.assert_called_once_with(
            {"query": "Apple grew 50%"}
        )
        assert rss_tool.search_rss_feeds.invoke.call_count == 3
        web_tool.invoke.assert_called_once_with({"query": "Apple grew 50%"})
        assert len(result["evidence"]) == 6

    @patch("graph.nodes.evidence_retriever._load_web_search_tool")
    @patch("graph.nodes.evidence_retriever._load_rss_tool")
    @patch("graph.nodes.evidence_retriever._load_reddit_tool")
    @patch("graph.nodes.evidence_retriever._load_news_tools")
    def test_handles_tool_errors_gracefully(
        self,
        mock_news_tools: MagicMock,
        mock_reddit_tool: MagicMock,
        mock_rss_tool: MagicMock,
        mock_web_tool: MagicMock,
    ):
        mock_news_tools.search_news.invoke.side_effect = Exception("boom")
        mock_reddit_tool.search_reddit.invoke.side_effect = Exception("boom")
        mock_rss_tool.search_rss_feeds.invoke.side_effect = Exception("boom")
        mock_web_tool.invoke.side_effect = Exception("boom")

        result = evidence_retriever(_state([Claim(id="c0", text="Test claim")]))
        assert "evidence" in result
        assert isinstance(result["evidence"], list)
        assert result["evidence"] == []

    @patch("graph.nodes.evidence_retriever._load_news_tools")
    def test_uses_first_claim_only(self, mock_news_tools: MagicMock):
        news_tools = MagicMock()
        mock_news_tools.return_value = news_tools
        news_tools.search_news.invoke.return_value = (
            "标题: First claim result\n链接: https://example.com/1"
        )

        result = evidence_retriever(
            _state(
                [
                    Claim(id="c0", text="First claim"),
                    Claim(id="c1", text="Second claim"),
                ]
            )
        )

        news_tools.search_news.invoke.assert_called_once_with({"query": "First claim"})
        assert any(item.claim_id == "c0" for item in result["evidence"])
        assert all(item.claim_id == "c0" for item in result["evidence"])

    def test_evidence_model_defaults_work(self):
        evidence = Evidence(claim_id="c0", content="Test", source_name="NewsAPI")
        assert evidence.stance == "neutral"
        assert evidence.credibility_score == 0.5
