import re
from collections.abc import Mapping
from typing import Literal, cast

from langchain_core.prompts import ChatPromptTemplate

Article = None

from graph.nodes.utils import emit_progress, invoke_structured, parse_json_response
from graph.state import Claim, FactCheckState


def _safe_emit_progress(stage: str, message: str) -> None:
    try:
        emit_progress(stage, message)
    except Exception:
        pass


def _detect_input_type(raw_input: str) -> str:
    stripped = raw_input.strip()
    if re.match(r"^https?://", stripped):
        return "url"
    if len(stripped) > 200 or (len(stripped) > 25 and len(stripped.split()) >= 5):
        return "text"
    return "claim"


def _fetch_url_content(url: str) -> str:
    try:
        article_cls = Article
        if article_cls is None:
            from newspaper import Article as article_cls

        article = article_cls(url)
        article.download()
        article.parse()
        return cast(str, article.text)
    except Exception as e:
        return f"[Error fetching URL: {e}]"


def claim_extractor(state: FactCheckState) -> dict[str, object]:
    raw_input = state.get("raw_input", "")
    language = state.get("language", "zh")

    _safe_emit_progress("claim_extraction", "Analyzing input...")

    state_input_type = state.get("input_type")
    input_type: Literal["url", "text", "claim"] = (
        state_input_type
        if state_input_type in {"url", "text", "claim"}
        else cast(Literal["url", "text", "claim"], _detect_input_type(raw_input))
    )

    content = raw_input
    if input_type == "url":
        _safe_emit_progress("claim_extraction", "Fetching content from URL...")
        content = _fetch_url_content(raw_input)

    if language == "zh":
        system_msg = """你是一个专业的声明提取专家。从给定的文本中提取所有可验证的核心声明。

规则：
1. 每个声明必须是可以被验证为真或假的具体断言
2. 排除主观观点和无法验证的表述
3. 保持声明的原始含义，不要改变措辞
4. 为每个声明分配唯一ID（claim_0, claim_1, ...）

请以JSON格式返回：
{{"claims": [{{"id": "claim_0", "text": "..."}}, {{"id": "claim_1", "text": "..."}}]}}"""
    else:
        system_msg = """You are a professional claim extraction expert. Extract all verifiable core claims from the given text.

Rules:
1. Each claim must be a specific assertion that can be verified as true or false
2. Exclude subjective opinions and unverifiable statements
3. Preserve the original meaning of each claim without changing wording
4. Assign unique IDs to each claim (claim_0, claim_1, ...)

Return in JSON format:
{{"claims": [{{"id": "claim_0", "text": "..."}}, {{"id": "claim_1", "text": "..."}}]}}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{content}"),
        ]
    )

    try:
        raw_response = invoke_structured(prompt, {}, {"content": content})
        parsed_json = parse_json_response(raw_response)
        parsed_claims: list[object] = []
        if isinstance(parsed_json, Mapping):
            claims_obj = parsed_json.get("claims", [])
            if isinstance(claims_obj, list):
                parsed_claims = claims_obj

        claims: list[Claim] = []
        for c in parsed_claims:
            if not isinstance(c, Mapping):
                continue
            claims.append(
                Claim(
                    id=str(c.get("id", f"claim_{len(claims)}")),
                    text=str(c.get("text", "")),
                    source_text=content[:200],
                )
            )

        if not claims:
            claims = [Claim(id="claim_0", text=raw_input, source_text=content[:200])]

        _safe_emit_progress("claim_extraction", f"Extracted {len(claims)} claims")

        return {
            "input_type": input_type,
            "claims": claims,
        }
    except Exception as e:
        return {
            "input_type": input_type,
            "claims": [Claim(id="claim_0", text=raw_input, source_text=raw_input[:200])],
            "errors": [f"claim_extractor error: {e}"],
        }
