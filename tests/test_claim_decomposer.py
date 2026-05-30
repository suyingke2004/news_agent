from unittest.mock import patch

from graph.nodes.claim_decomposer import _is_compound_claim, claim_decomposer
from graph.state import Claim


class TestIsCompoundClaim:
    def test_simple_claim(self):
        assert _is_compound_claim("Revenue grew 50%") is False

    def test_compound_with_and(self):
        assert _is_compound_claim("Revenue grew 50% and profits doubled") is True

    def test_compound_with_but(self):
        assert _is_compound_claim("Revenue grew but profits fell") is True

    def test_multiple_sentences(self):
        assert _is_compound_claim("Revenue grew. Profits fell.") is True

    def test_chinese_compound(self):
        assert _is_compound_claim("收入增长了50%而且利润翻倍了") is True

    def test_chinese_simple(self):
        assert _is_compound_claim("收入增长了50%") is False


class TestClaimDecomposer:
    def test_empty_claims(self):
        state = {"claims": [], "language": "en"}
        result = claim_decomposer(state)
        assert result["claims"] == []

    def test_simple_claims_pass_through(self):
        simple_claim = Claim(id="c0", text="Revenue grew 50%")
        state = {"claims": [simple_claim], "language": "en"}
        result = claim_decomposer(state)
        assert len(result["claims"]) == 1
        assert result["claims"][0].text == "Revenue grew 50%"

    @patch("graph.nodes.claim_decomposer.invoke_structured")
    def test_compound_claim_decomposed(self, mock_invoke):
        mock_invoke.return_value = '{"decomposed": [{"id": "c0_sub_0", "text": "Revenue grew 50%"}, {"id": "c0_sub_1", "text": "Profits doubled"}]}'

        compound = Claim(id="c0", text="Revenue grew 50% and profits doubled")
        state = {"claims": [compound], "language": "en"}
        result = claim_decomposer(state)

        assert len(result["claims"]) == 2
        assert result["claims"][0].text == "Revenue grew 50%"
        assert result["claims"][1].text == "Profits doubled"

    @patch("graph.nodes.claim_decomposer.invoke_structured")
    def test_exception_fallback(self, mock_invoke):
        mock_invoke.side_effect = Exception("LLM error")

        compound = Claim(id="c0", text="Revenue grew and profits doubled")
        state = {"claims": [compound], "language": "en"}
        result = claim_decomposer(state)

        assert len(result["claims"]) == 1
        assert result["claims"][0].text == "Revenue grew and profits doubled"

    @patch("graph.nodes.claim_decomposer.invoke_structured")
    def test_mixed_simple_and_compound(self, mock_invoke):
        mock_invoke.return_value = '{"decomposed": [{"id": "c1_sub_0", "text": "Part A"}, {"id": "c1_sub_1", "text": "Part B"}]}'

        simple = Claim(id="c0", text="Simple claim")
        compound = Claim(id="c1", text="Part A and Part B")
        state = {"claims": [simple, compound], "language": "en"}
        result = claim_decomposer(state)

        assert len(result["claims"]) == 3
