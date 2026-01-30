# Attack Vector Alpha F-002: Tokenization Mismatch Findings

## Executive Summary

Adversarial Unicode testing of `src/hf_gtc/preprocessing/tokenization.py` reveals multiple potential tokenizer divergence vectors. The `preprocess_text()` function does NOT normalize Unicode, leading to situations where visually identical inputs produce different outputs.

**Critical Finding**: NFC/NFD normalization divergence confirmed. The same visual character (e.g., "cafe") can have different byte representations that are NOT unified by preprocessing, causing downstream tokenizer divergence.

## Test Results Summary

| Category | Tests | Passed | Failed | Issues Found |
|----------|-------|--------|--------|--------------|
| ZWJ Sequences | 4 | 4 | 0 | Preserved (no normalization) |
| Bidi Overrides | 4 | 4 | 0 | **Preserved (security risk)** |
| Combining Marks | 3 | 2 | 1 | **NFC/NFD divergence** |
| Null/Control | 8 | 8 | 0 | **Preserved (potential injection)** |
| Surrogate Pairs | 3 | 3 | 0 | **Lone surrogates allowed** |
| Unassigned | 5 | 5 | 0 | Preserved |
| Homoglyphs | 3 | 3 | 0 | **Not normalized** |
| Whitespace | 7 | 7 | 0 | Partial normalization only |
| Normalization | 2 | 2 | 0 | **Divergence documented** |
| Hypothesis | 3 | 3 | 0 | No crashes |

## Critical Vulnerabilities

### 1. NFC/NFD Normalization Divergence (FAILED TEST)

**Location**: `test_combining_acute_accent`

**Evidence**:
```
NFC input:  'cafe' (U+00E9 - precomposed e-acute)  -> 'cafe'
NFD input:  'cafe' (U+0065 U+0301 - e + combining) -> 'cafe'

Result: DIFFERENT byte sequences despite identical visual appearance
```

**Impact**: A tokenizer will produce different token IDs for visually identical text:
- "cafe" (NFC) might tokenize to `[cafe]`
- "cafe" (NFD) might tokenize to `[caf, e, combining_mark]`

This causes:
- Model prediction inconsistency
- Potential adversarial attacks via normalization form manipulation
- Search/retrieval failures

### 2. Bidirectional Override Preservation (Security Risk)

**Evidence**:
```
Input:  'hello\u202edlrow'  (with RLO U+202E)
Output: 'hello\u202edlrow'  (RLO preserved)
```

**Impact**: Right-to-Left Override (U+202E) is preserved, enabling:
- Text spoofing attacks (displaying "safe" but executing "evil")
- Log injection
- UI-based attacks

### 3. Null Byte Preservation (Potential Injection)

**Evidence**:
```
Input:  'hello\x00world'
Output: 'hello\x00world'
```

**Impact**: Null bytes are NOT stripped, potentially causing:
- String truncation in C-based backends
- Injection attacks in systems that treat null as string terminator
- File path manipulation

### 4. Lone Surrogate Preservation (Invalid Unicode)

**Evidence**:
```
Input:  'hello\ud800world'  (lone high surrogate)
Output: 'hello\ud800world'
```

**Impact**: Invalid Unicode (lone surrogates U+D800-U+DFFF) is preserved:
- Can crash downstream systems expecting valid UTF-8
- May cause encoding errors when serializing to JSON/UTF-8

### 5. Homoglyph Non-Normalization

**Evidence**:
```
Latin 'a' (U+0061): 'hello abc'
Cyrillic 'a' (U+0430): 'hello abc'  <- DIFFERENT despite looking identical

Latin 'o' (U+006F): 'hello'
Greek 'o' (U+03BF): 'hello'  <- DIFFERENT despite looking identical
```

**Impact**: Homoglyph attacks are possible:
- Visually identical usernames/identifiers tokenize differently
- Adversarial examples can bypass filters
- Brand impersonation attacks

### 6. Fullwidth Character Non-Normalization

**Evidence**:
```
ASCII:     'hello'
Fullwidth: 'hello' (U+FF48 U+FF45 U+FF4C U+FF4C U+FF4F)

Result: Different outputs for visually similar text
```

### 7. Compatibility Decomposition Divergence

**Evidence**:
```
Ligature 'fi' (U+FB01): 'fi' (single character)
Separate 'fi': 'fi' (two characters)

Result: Different tokenization for semantically identical text
```

### 8. Zero-Width Characters Preserved

**Evidence**:
```
ZWSP (U+200B): 'hello\u200bworld' -> 'hello\u200bworld' (preserved)
Word Joiner (U+2060): preserved
BOM (U+FEFF): preserved in middle of string
```

**Impact**: Invisible characters can:
- Evade text filters
- Cause different tokenization
- Create "invisible" differences in identical-looking text

## Whitespace Normalization (Partial)

The following whitespace characters ARE normalized to regular space:
- No-Break Space (U+00A0)
- Ideographic Space (U+3000)
- En Quad (U+2000)
- Em Quad (U+2001)
- Line Separator (U+2028)
- Paragraph Separator (U+2029)

The following are NOT normalized:
- Zero-Width Space (U+200B)
- Word Joiner (U+2060)
- BOM (U+FEFF) in non-start position

## Recommended Mitigations (NOT IMPLEMENTED)

1. **Apply Unicode NFC normalization** before all preprocessing
2. **Strip or reject bidirectional override characters** (U+202A-U+202E, U+2066-U+2069)
3. **Strip null bytes** and other C0 control characters
4. **Reject lone surrogates** (U+D800-U+DFFF)
5. **Apply NFKC normalization** for compatibility character handling
6. **Strip zero-width characters** (U+200B, U+200C, U+200D except in emoji, U+2060, U+FEFF)
7. **Consider homoglyph detection** for security-sensitive applications

## Test File Location

All adversarial tests are in:
```
/home/noah/src/hf-ground-truth-corpus/tests/unit/test_tokenization_adversarial.py
```

Run with:
```bash
uv run pytest tests/unit/test_tokenization_adversarial.py -v -s
```

## Conclusion

The `preprocess_text()` function in `tokenization.py` provides minimal normalization (lowercase, whitespace collapsing) but does NOT address Unicode normalization, control characters, or homoglyphs. This creates multiple attack vectors for adversarial inputs that could cause tokenizer divergence between:

1. Training and inference pipelines
2. Different tokenizer implementations
3. User input and stored/indexed text

These findings document potential security and correctness issues that should be addressed in the preprocessing pipeline.

---
*Generated by Attack Vector Alpha F-002 analysis*
*Date: 2025-01-30*
