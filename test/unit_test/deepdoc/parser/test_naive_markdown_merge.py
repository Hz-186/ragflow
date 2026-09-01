"""
Unit tests for markdown chunk merging logic in rag/app/naive.py.

Tests the _is_short_header() helper function to ensure short markdown headers
are correctly identified and will be force-merged with the next section.

Uses lazy import via fixture to avoid triggering deepdoc model loading
at pytest collection time (which would fail in CI without model files).
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[4]


class TestIsShortHeader:
    """Test cases for _is_short_header() function."""

    @pytest.fixture(autouse=True)
    def _lazy_import(self):
        sys.path.insert(0, str(_REPO))
        from rag.app.naive import _is_short_header

        self._is_short_header = _is_short_header

    def test_short_header_h1(self):
        """Short level-1 header should return True."""
        text = "# Quick Start"
        result = self._is_short_header(text)
        assert result is True

    def test_short_header_h2(self):
        """Short level-2 header should return True."""
        text = "## Quick Travel"
        result = self._is_short_header(text)
        assert result is True

    def test_short_header_h3(self):
        """Short level-3 header should return True."""
        text = "### Setup"
        result = self._is_short_header(text)
        assert result is True

    def test_long_header(self):
        """Long header (> 50 tokens) should return False."""
        text = "# " + "Very long header " * 20  # ~100 tokens
        result = self._is_short_header(text)
        assert result is False

    def test_non_header_short_text(self):
        """Short text without header pattern should return False."""
        text = "This is short"
        result = self._is_short_header(text)
        assert result is False

    def test_empty_text(self):
        """Empty text should return False."""
        text = ""
        result = self._is_short_header(text)
        assert result is False

    def test_whitespace_only(self):
        """Whitespace-only text should return False."""
        text = "   "
        result = self._is_short_header(text)
        assert result is False

    def test_header_exactly_50_tokens(self):
        """Header with exactly 50 tokens should return False (strict <)."""
        words = ["word"] * 49
        text = "# " + " ".join(words)
        result = self._is_short_header(text, max_tokens=50)
        assert result is False

    def test_header_49_tokens(self):
        """Header with 49 tokens should return True (< 50)."""
        words = ["word"] * 48
        text = "# " + " ".join(words)
        result = self._is_short_header(text, max_tokens=50)
        assert result is True

    def test_custom_max_tokens(self):
        """Should respect custom max_tokens parameter."""
        # "# Short" = 2 tokens in cl100k_base encoding
        text = "# Short"
        result = self._is_short_header(text, max_tokens=5)
        assert result is True  # 2 < 5 → short

        result = self._is_short_header(text, max_tokens=2)
        assert result is False  # 2 < 2 → not short

    def test_header_with_special_chars(self):
        """Header with special characters should still be recognized."""
        text = "## API Endpoint: /api/v1/users"
        result = self._is_short_header(text)
        assert result is True

    def test_header_with_cjk_chars(self):
        """Header with CJK characters should be recognized."""
        text = "## 快速旅行"
        result = self._is_short_header(text)
        assert result is True


class TestNormalizeSectionTextForRtlPresentationForms:
    """Test cases for _normalize_section_text_for_rtl_presentation_forms() function."""

    @pytest.fixture(autouse=True)
    def _lazy_import(self):
        sys.path.insert(0, str(_REPO))
        from rag.app.naive import _normalize_section_text_for_rtl_presentation_forms

        self._normalize = _normalize_section_text_for_rtl_presentation_forms

    def test_empty_or_none(self):
        assert self._normalize(None) is None
        assert self._normalize([]) == []

    def test_tuple_sections(self):
        # \uFE8D is ARABIC LETTER ALEF ISOLATED FORM -> \u0627
        # \uFE91 is ARABIC LETTER BEH INITIAL FORM -> \u0628
        # \uFEEC is ARABIC LETTER HEH INITIAL FORM -> \u0647
        sections = [("\uFE8D\uFE91\uFEEC", "extra_meta_1", 123), ()]
        res = self._normalize(sections)
        assert len(res) == 2
        assert res[0][0] == "\u0627\u0628\u0647"
        assert res[0][1:] == ("extra_meta_1", 123)
        assert res[1] == ()

    def test_list_sections(self):
        sections = [["\uFE8D\uFE91", "meta_a"], []]
        res = self._normalize(sections)
        assert len(res) == 2
        assert res[0][0] == "\u0627\u0628"
        assert res[0][1:] == ["meta_a"]
        assert res[1] == []

    def test_string_sections(self):
        sections = ["\uFE8D\uFE91", "Normal text"]
        res = self._normalize(sections)
        assert res == ["\u0627\u0628", "Normal text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
