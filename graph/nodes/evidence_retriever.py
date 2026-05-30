import re
from importlib import import_module
from typing import Protocol, cast

from graph.nodes.utils import emit_progress
from graph.state import Evidence, FactCheckState


RSS_FEED_SOURCES = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "Reuters": "https://www.reuters.com/news/rss.xml",
}


class InvokableTool(Protocol):
    def invoke(self, payload: dict[str, str]) -> str: ...


def _safe_emit_progress(stage: str, message: str) -> None:
    try:
        emit_progress(stage, message)
    except Exception:
        pass


def _tool_attr(instance: object, name: str) -> InvokableTool:
    return cast(InvokableTool, getattr(instance, name))


def _invoke_tool(tool_target: InvokableTool, payload: dict[str, str]) -> str:
    return tool_target.invoke(payload)


def _load_news_tools() -> object | None:
    try:
        return import_module("tools").news_tools
    except Exception:
        return None


def _load_reddit_tool() -> object | None:
    try:
        return import_module("tools.reddit_search").reddit_search_tool
    except Exception:
        return None


def _load_rss_tool() -> object | None:
    try:
        return import_module("tools.rss_feed").rss_feed_tool
    except Exception:
        return None


def _load_web_search_tool() -> InvokableTool | None:
    try:
        return cast(InvokableTool, import_module("tools.news_website_search").search_news_websites)
    except Exception:
        return None


def _parse_news_results(raw_text: str, claim_id: str) -> list[Evidence]:
    if not raw_text or "没有找到" in raw_text or "出错" in raw_text:
        return []

    evidences: list[Evidence] = []
    entries = raw_text.split("\n\n")
    for entry in entries:
        if not entry.strip():
            continue
        title_match = re.search(r"标题:\s*(.+)", entry)
        link_match = re.search(r"链接:\s*(.+)", entry)
        if title_match:
            evidences.append(
                Evidence(
                    claim_id=claim_id,
                    content=title_match.group(1).strip(),
                    source_url=link_match.group(1).strip() if link_match else "",
                    source_name="NewsAPI",
                    source_type="news",
                )
            )
    return evidences


def _parse_reddit_results(raw_text: str, claim_id: str) -> list[Evidence]:
    if not raw_text or "没有找到" in raw_text or "error" in raw_text.lower() or "出错" in raw_text:
        return []

    evidences: list[Evidence] = []
    entries = raw_text.split("\n\n")
    for entry in entries:
        if not entry.strip():
            continue
        title_match = re.search(r"标题:\s*(.+)", entry)
        link_match = re.search(r"链接:\s*(.+)", entry)
        score_match = re.search(r"评分:\s*(.+)", entry)
        content_parts: list[str] = []
        if title_match:
            content_parts.append(f"标题: {title_match.group(1).strip()}")
        if score_match:
            content_parts.append(f"评分: {score_match.group(1).strip()}")
        if content_parts:
            evidences.append(
                Evidence(
                    claim_id=claim_id,
                    content="\n".join(content_parts),
                    source_url=link_match.group(1).strip() if link_match else "",
                    source_name="Reddit",
                    source_type="reddit",
                )
            )
    return evidences[:3]


def _parse_rss_results(raw_text: str, claim_id: str, source_name: str = "RSS Feed") -> list[Evidence]:
    if (
        not raw_text
        or "没有找到" in raw_text
        or "不能为空" in raw_text
        or "出错" in raw_text
        or "没有条目" in raw_text
    ):
        return []

    evidences: list[Evidence] = []
    entries = raw_text.split("\n\n")
    for entry in entries:
        if not entry.strip() or entry.startswith("RSS订阅源:"):
            continue
        title_match = re.search(r"标题:\s*(.+)", entry)
        link_match = re.search(r"链接:\s*(.+)", entry)
        if title_match:
            evidences.append(
                Evidence(
                    claim_id=claim_id,
                    content=title_match.group(1).strip(),
                    source_url=link_match.group(1).strip() if link_match else "",
                    source_name=source_name,
                    source_type="rss",
                )
            )
    return evidences[:5]


def _parse_web_results(raw_text: str, claim_id: str) -> list[Evidence]:
    if not raw_text or "未在" in raw_text or "出错" in raw_text:
        return []

    evidences: list[Evidence] = []
    entries = raw_text.split("\n\n")
    for entry in entries:
        if not entry.strip() or entry.startswith("关于'"):
            continue
        source_match = re.search(r"来源:\s*(.+)", entry)
        title_match = re.search(r"标题:\s*(.+)", entry)
        link_match = re.search(r"链接:\s*(.+)", entry)
        if title_match:
            evidences.append(
                Evidence(
                    claim_id=claim_id,
                    content=title_match.group(1).strip(),
                    source_url=link_match.group(1).strip() if link_match else "",
                    source_name=source_match.group(1).strip() if source_match else "Web",
                    source_type="web",
                )
            )
    return evidences[:5]


def evidence_retriever(state: FactCheckState) -> dict[str, list[Evidence]]:
    claims = state["claims"] if "claims" in state else []
    if not claims:
        return {"evidence": []}

    claim = claims[0]
    _safe_emit_progress(
        "evidence_retrieval", f"Searching sources for: {claim.text[:50]}..."
    )

    all_evidence: list[Evidence] = []

    news_tools = _load_news_tools()
    if news_tools is not None:
        try:
            result = _invoke_tool(_tool_attr(news_tools, "search_news"), {"query": claim.text})
            all_evidence.extend(_parse_news_results(result, claim.id))
        except Exception:
            pass

    reddit_tool = _load_reddit_tool()
    if reddit_tool is not None:
        try:
            result = _invoke_tool(
                _tool_attr(reddit_tool, "search_reddit"), {"query": claim.text}
            )
            all_evidence.extend(_parse_reddit_results(result, claim.id))
        except Exception:
            pass

    rss_tool = _load_rss_tool()
    if rss_tool is not None:
        for source_name, feed_url in RSS_FEED_SOURCES.items():
            try:
                result = _invoke_tool(
                    _tool_attr(rss_tool, "search_rss_feeds"), {"feed_url": feed_url}
                )
                rss_evidence = _parse_rss_results(result, claim.id, source_name=source_name)
                filtered = [
                    item
                    for item in rss_evidence
                    if claim.text.lower() in item.content.lower()
                    or claim.text.lower() in item.source_url.lower()
                ]
                all_evidence.extend(filtered[:2] if filtered else rss_evidence[:1])
            except Exception:
                pass

    web_search_tool = _load_web_search_tool()
    if web_search_tool is not None:
        try:
            result = _invoke_tool(web_search_tool, {"query": claim.text})
            all_evidence.extend(_parse_web_results(result, claim.id))
        except Exception:
            pass

    _safe_emit_progress(
        "evidence_retrieval",
        f"Found {len(all_evidence)} evidence items for claim {claim.id}",
    )
    return {"evidence": all_evidence}
