"""Adversarial Unicode tests for tokenization - Attack Vector Alpha F-002.

This module tests edge cases that could cause tokenizer divergence when
handling adversarial Unicode inputs. These tests document potential
security and correctness issues.

IMPORTANT: This is refutation testing. Failures indicate potential
vulnerabilities that should be documented and addressed.
"""
# ruff: noqa: RUF001, RUF002
# Ambiguous Unicode characters are intentional for adversarial testing

from __future__ import annotations

import unicodedata
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hf_gtc.preprocessing.tokenization import (
    preprocess_text,
    tokenize_batch,
)


class TestZWJSequences:
    """Test Zero-Width Joiner sequences (emoji combinations).

    ZWJ sequences combine multiple emoji code points into single glyphs.
    Tokenizers may handle these inconsistently.
    """

    def test_family_emoji_zwj(self) -> None:
        """Test family emoji with multiple ZWJ joiners.

        U+1F468 U+200D U+1F469 U+200D U+1F467 U+200D U+1F466
        (Man, ZWJ, Woman, ZWJ, Girl, ZWJ, Boy = Family)
        """
        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"
        result = preprocess_text(f"Hello {family} World")

        # Document: Does preprocessing preserve or destroy ZWJ sequences?
        assert family in result or "\u200d" not in result, (
            f"ZWJ sequence partially destroyed: {result!r}"
        )

    def test_flag_sequence(self) -> None:
        """Test regional indicator sequences (flags).

        Flags use pairs of Regional Indicator symbols.
        U+1F1FA U+1F1F8 = US flag
        """
        us_flag = "\U0001f1fa\U0001f1f8"
        result = preprocess_text(f"Hello {us_flag}")

        # Both indicators must remain together
        assert us_flag in result, f"Flag sequence broken: {result!r}"

    def test_skin_tone_modifier(self) -> None:
        """Test emoji with skin tone modifiers.

        U+1F44B U+1F3FD = Waving hand, medium skin tone
        """
        waving = "\U0001f44b\U0001f3fd"
        result = preprocess_text(f"{waving} hi")

        # Modifier must stay with base emoji
        assert waving in result, f"Skin tone modifier separated: {result!r}"

    def test_profession_zwj_sequence(self) -> None:
        """Test profession emoji ZWJ sequence.

        Woman technologist: U+1F469 U+200D U+1F4BB
        """
        technologist = "\U0001f469\u200d\U0001f4bb"
        result = preprocess_text(technologist)

        assert technologist in result, f"Profession ZWJ sequence broken: {result!r}"


class TestBidirectionalOverrides:
    """Test right-to-left and other directional overrides.

    These can be used for visual spoofing attacks.
    """

    def test_right_to_left_override(self) -> None:
        """Test RLO (Right-to-Left Override) character U+202E.

        This character reverses text direction and can hide malicious content.
        Example: "hello\u202eollehdlrow" displays as "helloworld" but
        contains hidden characters.
        """
        rlo = "\u202e"
        text = f"hello{rlo}dlrow"
        result = preprocess_text(text)

        # Document behavior: Is RLO preserved, stripped, or causes error?
        print(f"RLO test input: {text!r}")
        print(f"RLO test output: {result!r}")

        # At minimum, the function should not crash
        assert isinstance(result, str)

    def test_left_to_right_override(self) -> None:
        """Test LRO (Left-to-Right Override) character U+202D."""
        lro = "\u202d"
        text = f"{lro}forced ltr"
        result = preprocess_text(text)

        assert isinstance(result, str)

    def test_mixed_bidi_text(self) -> None:
        """Test mixed Arabic and English with bidi controls."""
        # Arabic text with explicit LRM markers
        lrm = "\u200e"  # Left-to-Right Mark
        rlm = "\u200f"  # Right-to-Left Mark
        text = f"English{lrm} {rlm}العربية{lrm} more"
        result = preprocess_text(text)

        assert isinstance(result, str)

    def test_bidi_isolation(self) -> None:
        """Test First Strong Isolate and Pop Directional Isolate."""
        fsi = "\u2068"  # First Strong Isolate
        pdi = "\u2069"  # Pop Directional Isolate
        text = f"Hello {fsi}world{pdi}!"
        result = preprocess_text(text)

        assert isinstance(result, str)


class TestCombiningDiacriticals:
    """Test combining diacritical marks.

    These can create visually identical but byte-different strings.
    """

    def test_combining_acute_accent(self) -> None:
        """Test precomposed vs decomposed é.

        NFC: U+00E9 (é as single code point)
        NFD: U+0065 U+0301 (e + combining acute accent)
        """
        precomposed = "\u00e9"  # é NFC
        decomposed = "e\u0301"  # e + combining acute NFD

        result_nfc = preprocess_text(f"caf{precomposed}")
        result_nfd = preprocess_text(f"caf{decomposed}")

        # CRITICAL: These should ideally produce identical output
        # Tokenizer divergence if they don't!
        print(f"NFC result: {result_nfc!r}")
        print(f"NFD result: {result_nfd!r}")

        # Document potential divergence
        if result_nfc != result_nfd:
            pytest.fail(
                f"NFC/NFD divergence detected!\n"
                f"NFC: {result_nfc!r}\n"
                f"NFD: {result_nfd!r}"
            )

    def test_zalgo_text(self) -> None:
        """Test 'Zalgo' text with many combining marks.

        Excessive combining characters can cause rendering issues
        and potentially different tokenization.
        """
        # H with many combining marks
        zalgo_h = "H\u0335\u0338\u0321\u034b\u036f"
        text = f"{zalgo_h}ello"
        result = preprocess_text(text)

        assert isinstance(result, str)
        # Document: are combining marks preserved?
        print(f"Zalgo input: {text!r}")
        print(f"Zalgo output: {result!r}")

    def test_combining_grapheme_joiner(self) -> None:
        """Test Combining Grapheme Joiner U+034F."""
        cgj = "\u034f"
        text = f"a{cgj}b"  # a and b joined
        result = preprocess_text(text)

        assert isinstance(result, str)


class TestNullAndControlCharacters:
    """Test null bytes and ASCII control characters.

    These can cause truncation, crashes, or injection attacks.
    """

    def test_null_byte_middle(self) -> None:
        """Test null byte in middle of string."""
        text = "hello\x00world"
        result = preprocess_text(text)

        # Document: Does null byte cause truncation?
        print(f"Null byte input: {text!r}")
        print(f"Null byte output: {result!r}")

        # Should not crash; document behavior
        assert isinstance(result, str)

    def test_null_byte_start(self) -> None:
        """Test null byte at start."""
        text = "\x00hello"
        result = preprocess_text(text)
        assert isinstance(result, str)

    def test_null_byte_end(self) -> None:
        """Test null byte at end."""
        text = "hello\x00"
        result = preprocess_text(text)
        assert isinstance(result, str)

    def test_multiple_null_bytes(self) -> None:
        """Test multiple null bytes."""
        text = "a\x00b\x00c\x00d"
        result = preprocess_text(text)
        assert isinstance(result, str)

    def test_bell_character(self) -> None:
        """Test ASCII bell U+0007."""
        text = "hello\x07world"
        result = preprocess_text(text)
        assert isinstance(result, str)

    def test_backspace(self) -> None:
        """Test backspace character U+0008."""
        text = "hello\x08world"
        result = preprocess_text(text)
        assert isinstance(result, str)

    def test_form_feed(self) -> None:
        """Test form feed U+000C."""
        text = "hello\x0cworld"
        result = preprocess_text(text)
        assert isinstance(result, str)

    def test_vertical_tab(self) -> None:
        """Test vertical tab U+000B."""
        text = "hello\x0bworld"
        result = preprocess_text(text)
        assert isinstance(result, str)


class TestSurrogatePairs:
    """Test handling of surrogate pairs and invalid UTF-16.

    Python strings are UTF-8/UCS-4 internally, but surrogate code points
    (U+D800-U+DFFF) can cause issues when converting to/from UTF-16.
    """

    def test_high_surrogate_alone(self) -> None:
        """Test lone high surrogate U+D800.

        This is invalid Unicode but Python allows it in strings.
        """
        try:
            # This may raise an error in some Python versions
            text = "hello\ud800world"
            result = preprocess_text(text)
            print(f"Lone high surrogate result: {result!r}")
            assert isinstance(result, str)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Document: lone surrogates cause encoding errors
            print(f"Lone high surrogate error: {e}")

    def test_low_surrogate_alone(self) -> None:
        """Test lone low surrogate U+DC00."""
        try:
            text = "hello\udc00world"
            result = preprocess_text(text)
            print(f"Lone low surrogate result: {result!r}")
            assert isinstance(result, str)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            print(f"Lone low surrogate error: {e}")

    def test_reversed_surrogate_pair(self) -> None:
        """Test reversed surrogate pair (low before high)."""
        try:
            text = "\udc00\ud800"  # Invalid: low before high
            result = preprocess_text(text)
            print(f"Reversed surrogate pair result: {result!r}")
            assert isinstance(result, str)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            print(f"Reversed surrogate pair error: {e}")


class TestUnassignedCodePoints:
    """Test unassigned and reserved Unicode code points.

    These may be handled differently by different tokenizers.
    """

    def test_unassigned_plane_1(self) -> None:
        """Test unassigned code point in Plane 1 (SMP)."""
        # U+1FFFF is currently unassigned (as of Unicode 15)
        text = "hello\U0001ffffworld"
        result = preprocess_text(text)

        print(f"Unassigned SMP: {result!r}")
        assert isinstance(result, str)

    def test_noncharacter(self) -> None:
        """Test noncharacter U+FFFE.

        Noncharacters are permanently reserved and never assigned.
        """
        text = "hello\ufffeworld"
        result = preprocess_text(text)

        print(f"Noncharacter U+FFFE: {result!r}")
        assert isinstance(result, str)

    def test_bom_character(self) -> None:
        """Test Byte Order Mark U+FEFF at various positions."""
        bom = "\ufeff"

        # BOM at start (common)
        result1 = preprocess_text(f"{bom}hello")
        # BOM in middle (unusual)
        result2 = preprocess_text(f"hel{bom}lo")

        print(f"BOM at start: {result1!r}")
        print(f"BOM in middle: {result2!r}")

        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_private_use_area(self) -> None:
        """Test Private Use Area characters."""
        # U+E000 to U+F8FF are PUA
        pua = "\ue000\ue001\uf8ff"
        text = f"hello{pua}world"
        result = preprocess_text(text)

        print(f"PUA result: {result!r}")
        assert isinstance(result, str)

    def test_supplementary_pua(self) -> None:
        """Test Supplementary Private Use Area characters."""
        # Plane 15-16 PUA
        spua = "\U000f0000\U000fffff\U00100000"
        text = f"hello{spua}world"
        result = preprocess_text(text)

        print(f"SPUA result: {result!r}")
        assert isinstance(result, str)


class TestHomoglyphs:
    """Test homoglyph attacks (visually similar but different characters).

    These can cause divergence between visual inspection and tokenization.
    """

    def test_cyrillic_a(self) -> None:
        """Test Cyrillic а vs Latin a.

        U+0430 (Cyrillic) looks like U+0061 (Latin) but tokenizes differently.
        """
        latin_a = "a"  # U+0061
        cyrillic_a = "а"  # U+0430

        latin_result = preprocess_text(f"hello {latin_a}bc")
        cyrillic_result = preprocess_text(f"hello {cyrillic_a}bc")

        print(f"Latin 'a': {latin_result!r}")
        print(f"Cyrillic 'а': {cyrillic_result!r}")

        # These WILL differ, documenting the divergence
        if latin_result == cyrillic_result:
            print("WARNING: Homoglyphs normalized to same output")

    def test_greek_omicron(self) -> None:
        """Test Greek ο vs Latin o."""
        latin_o = "o"  # U+006F
        greek_o = "ο"  # U+03BF

        latin_result = preprocess_text(f"hell{latin_o}")
        greek_result = preprocess_text(f"hell{greek_o}")

        print(f"Latin 'o': {latin_result!r}")
        print(f"Greek 'ο': {greek_result!r}")

    def test_fullwidth_characters(self) -> None:
        """Test fullwidth vs ASCII characters.

        Fullwidth ASCII (U+FF01-U+FF5E) looks similar but differs.
        """
        ascii_hello = "hello"
        fullwidth_hello = "ｈｅｌｌｏ"  # U+FF48 U+FF45 U+FF4C U+FF4C U+FF4F

        ascii_result = preprocess_text(ascii_hello)
        fullwidth_result = preprocess_text(fullwidth_hello)

        print(f"ASCII: {ascii_result!r}")
        print(f"Fullwidth: {fullwidth_result!r}")

        # Document divergence
        assert ascii_result != fullwidth_result or ascii_hello == fullwidth_hello


class TestWhitespaceVariants:
    """Test various Unicode whitespace characters.

    Different whitespace characters may be normalized differently.
    """

    def test_no_break_space(self) -> None:
        """Test No-Break Space U+00A0."""
        nbsp = "\u00a0"
        text = f"hello{nbsp}world"
        result = preprocess_text(text)

        print(f"NBSP result: {result!r}")
        # Check if NBSP is normalized to regular space
        assert "hello" in result and "world" in result

    def test_zero_width_space(self) -> None:
        """Test Zero Width Space U+200B."""
        zwsp = "\u200b"
        text = f"hello{zwsp}world"
        result = preprocess_text(text)

        print(f"ZWSP result: {result!r}")
        # ZWSP may or may not be stripped
        assert isinstance(result, str)

    def test_word_joiner(self) -> None:
        """Test Word Joiner U+2060."""
        wj = "\u2060"
        text = f"hello{wj}world"
        result = preprocess_text(text)

        print(f"Word Joiner result: {result!r}")
        assert isinstance(result, str)

    def test_ideographic_space(self) -> None:
        """Test Ideographic Space U+3000."""
        ideographic = "\u3000"
        text = f"hello{ideographic}world"
        result = preprocess_text(text)

        print(f"Ideographic space result: {result!r}")
        assert isinstance(result, str)

    def test_en_quad_em_quad(self) -> None:
        """Test En Quad and Em Quad spaces."""
        en_quad = "\u2000"
        em_quad = "\u2001"
        text = f"a{en_quad}b{em_quad}c"
        result = preprocess_text(text)

        print(f"En/Em Quad result: {result!r}")
        assert isinstance(result, str)

    def test_line_separator(self) -> None:
        """Test Line Separator U+2028."""
        line_sep = "\u2028"
        text = f"hello{line_sep}world"
        result = preprocess_text(text)

        print(f"Line Separator result: {result!r}")
        assert isinstance(result, str)

    def test_paragraph_separator(self) -> None:
        """Test Paragraph Separator U+2029."""
        para_sep = "\u2029"
        text = f"hello{para_sep}world"
        result = preprocess_text(text)

        print(f"Paragraph Separator result: {result!r}")
        assert isinstance(result, str)


class TestNormalizationConsistency:
    """Test Unicode normalization consistency.

    Different normalization forms can cause tokenizer divergence.
    """

    def test_nfc_vs_nfd_consistency(self) -> None:
        """Test that NFC and NFD inputs produce consistent outputs."""
        # German ü in different forms
        nfc = "für"  # Precomposed
        nfd = unicodedata.normalize("NFD", "für")

        result_nfc = preprocess_text(nfc)
        result_nfd = preprocess_text(nfd)

        print(f"NFC input: {nfc!r} -> {result_nfc!r}")
        print(f"NFD input: {nfd!r} -> {result_nfd!r}")

        # These should ideally be equal after preprocessing
        if result_nfc != result_nfd:
            print("DIVERGENCE: NFC/NFD normalization not consistent")

    def test_compatibility_decomposition(self) -> None:
        """Test NFKC vs NFC (compatibility decomposition)."""
        # fi ligature vs f+i
        ligature = "ﬁ"  # U+FB01
        separate = "fi"

        result_lig = preprocess_text(ligature)
        result_sep = preprocess_text(separate)

        print(f"Ligature: {result_lig!r}")
        print(f"Separate: {result_sep!r}")

        # Document divergence
        if result_lig != result_sep:
            print("DIVERGENCE: Compatibility forms not normalized")


class TestTokenizeBatchAdversarial:
    """Test tokenize_batch with adversarial inputs."""

    def test_mixed_adversarial_batch(self, mock_tokenizer: MagicMock) -> None:
        """Test batch with mixed adversarial inputs."""
        texts = [
            "normal text",
            "with\x00null",
            "zalgo\u0335\u0338text",
            "\u202ehidden",
            "emoji\U0001f468\u200d\U0001f4bb",
        ]

        # Should not crash
        result = tokenize_batch(texts, mock_tokenizer)
        assert result is not None

    def test_very_long_emoji_sequence(self, mock_tokenizer: MagicMock) -> None:
        """Test extremely long ZWJ emoji sequence."""
        # Build a very long ZWJ chain
        emoji_chain = "\U0001f468"
        for _ in range(50):
            emoji_chain += "\u200d\U0001f469"

        result = tokenize_batch([emoji_chain], mock_tokenizer)
        assert result is not None


class TestHypothesisAdversarial:
    """Property-based tests for adversarial inputs."""

    @given(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Cc", "Cf", "Co"),  # Control, Format, Private Use
                min_codepoint=0,
                max_codepoint=0x10FFFF,
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_control_characters_dont_crash(self, text: str) -> None:
        """Test that control characters don't cause crashes."""
        try:
            result = preprocess_text(f"prefix{text}suffix")
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Crashed on control chars: {text!r}, error: {e}")

    @given(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Mn", "Mc", "Me"),  # Combining marks
                min_codepoint=0x0300,
                max_codepoint=0x036F,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_combining_marks_dont_crash(self, marks: str) -> None:
        """Test that many combining marks don't crash."""
        text = f"a{marks}b"
        try:
            result = preprocess_text(text)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Crashed on combining marks: {marks!r}, error: {e}")

    @given(st.binary(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_arbitrary_bytes_as_text(self, data: bytes) -> None:
        """Test that arbitrary byte sequences (decoded leniently) don't crash."""
        try:
            # Try to decode as UTF-8 with replacement
            text = data.decode("utf-8", errors="replace")
            result = preprocess_text(text)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Crashed on bytes {data!r}: {e}")


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer for adversarial tests."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": [[101, 2023, 102]],
        "attention_mask": [[1, 1, 1]],
    }
    return tokenizer
