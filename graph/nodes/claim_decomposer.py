from langchain_core.prompts import ChatPromptTemplate

from graph.nodes.utils import emit_progress, invoke_structured, parse_json_response
from graph.state import Claim, FactCheckState


def _is_compound_claim(text: str) -> bool:
    if text.count(".") > 1 or text.count("。") > 1:
        return True

    compound_indicators = [
        " and ",
        " but ",
        " however ",
        " although ",
        " while ",
        "而且",
        "但是",
        "然而",
        "同时",
        "并且",
        "虽然",
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in compound_indicators)


def claim_decomposer(state: FactCheckState) -> dict:
    claims = state.get("claims", [])
    language = state.get("language", "zh")

    if not claims:
        return {"claims": []}

    emit_progress("claim_decomposition", f"Decomposing {len(claims)} claims...")

    decomposed = []
    for claim in claims:
        if not _is_compound_claim(claim.text):
            decomposed.append(claim)
            continue

        if language == "zh":
            system_msg = "你是一个声明分解专家。将复合声明分解为独立的、可验证的原子声明。\n\n规则：\n1. 每个原子声明必须是独立的、可以被验证为真或假的具体断言\n2. 保持原始含义不变\n3. 使用原始声明的ID作为前缀，如 claim_0 -> claim_0_sub_0, claim_0_sub_1\n4. 如果声明已经是原子性的，直接返回原声明\n\n请以JSON格式返回，包含一个decomposed数组，每个元素有id和text字段。"
        else:
            system_msg = "You are a claim decomposition expert. Break compound claims into independent, verifiable atomic sub-claims.\n\nRules:\n1. Each atomic claim must be an independent assertion that can be verified as true or false\n2. Preserve the original meaning\n3. Use the original claim ID as prefix, e.g. claim_0 -> claim_0_sub_0, claim_0_sub_1\n4. If the claim is already atomic, return it unchanged\n\nReturn JSON with a 'decomposed' array where each element has 'id' and 'text' fields."

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "Decompose this claim:\n{claim_text}\nClaim ID: {claim_id}"),
            ]
        )

        try:
            raw = invoke_structured(prompt, {}, {"claim_text": claim.text, "claim_id": claim.id})
            parsed = parse_json_response(raw)

            for sub in parsed.get("decomposed", []):
                decomposed.append(
                    Claim(
                        id=sub.get("id", f"{claim.id}_sub_{len(decomposed)}"),
                        text=sub.get("text", claim.text),
                        source_text=claim.source_text,
                    )
                )
        except Exception:
            decomposed.append(claim)

    emit_progress(
        "claim_decomposition", f"Decomposed into {len(decomposed)} atomic claims"
    )
    return {"claims": decomposed}
