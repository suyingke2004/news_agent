"""Edge case tests for claim_decomposer node."""
from unittest.mock import MagicMock, patch
from typing import cast

from graph.nodes.claim_decomposer import claim_decomposer, _is_compound_claim
from graph.state import Claim


class TestIsCompoundClaim:
    def test_simple_claim_is_not_compound(self):
        assert _is_compound_claim("The Earth is flat") is False

    def test_compound_with_and(self):
        assert _is_compound_claim("The Earth is flat and the Moon is made of cheese") is True

    def test_compound_with_but(self):
        assert _is_compound_claim("AI will replace jobs but create new ones") is True

    def test_compound_chinese_with_and(self):
        assert _is_compound_claim("人工智能将取代工作岗位并且创造新机会") is True

    def test_short_chinese_not_compound(self):
        assert _is_compound_claim("这是一个声明") is False

    def test_empty_string_not_compound(self):
        assert _is_compound_claim("") is False


class TestClaimDecomposerEdgeCases:
    def test_empty_claims_returns_empty(self):
        """Empty claims list should return empty claims."""
        result = claim_decomposer({"claims": [], "language": "en"})
        assert result["claims"] == []

    def test_simple_claim_passes_through(self):
        """Non-compound claim should pass through unchanged."""
        simple_claim = Claim(id="c0", text="The Earth is flat", source_text="test")
        result = claim_decomposer({"claims": [simple_claim], "language": "en"})
        claims = cast(list[Claim], result["claims"])
        # Simple claims should pass through without decomposition
        assert len(claims) >= 1
        assert claims[0].text == "The Earth is flat"

    @patch("graph.nodes.claim_decomposer.invoke_structured")
    @patch("graph.nodes.claim_decomposer._is_compound_claim")
    def test_compound_claim_decomposed(self, mock_is_compound: MagicMock, mock_invoke: MagicMock):
        """Compound claim should be decomposed via LLM."""
        mock_is_compound.return_value = True
        mock_invoke.return_value = '{"decomposed": [{"id": "c0_0", "text": "The Earth is flat"}, {"id": "c0_1", "text": "The Moon is made of cheese"}]}'

        compound = Claim(id="c0", text="The Earth is flat and the Moon is made of cheese and the Sun orbits the Earth", source_text="test")
        result = claim_decomposer({"claims": [compound], "language": "en"})
        claims = cast(list[Claim], result["claims"])
        assert len(claims) == 2

    @patch("graph.nodes.claim_decomposer.invoke_structured")
    @patch("graph.nodes.claim_decomposer._is_compound_claim")
    def test_llm_failure_falls_back_to_original(self, mock_is_compound: MagicMock, mock_invoke: MagicMock):
        """When LLM fails during decomposition, fall back to original claim."""
        mock_is_compound.return_value = True
        mock_invoke.side_effect = Exception("LLM error")

        compound = Claim(id="c0", text="A and B and C", source_text="test")
        result = claim_decomposer({"claims": [compound], "language": "en"})
        claims = cast(list[Claim], result["claims"])
        assert len(claims) == 1
        assert claims[0].text == "A and B and C"
