from app.rag.cleaning import clean_text


def test_collapses_excess_blank_lines():
    raw = "Para one.\n\n\n\n\nPara two."
    result = clean_text(raw)
    assert "\n\n\n" not in result
    assert "Para one." in result and "Para two." in result


def test_collapses_excess_spaces():
    raw = "Word1     Word2"
    assert clean_text(raw) == "Word1 Word2"


def test_strips_control_characters():
    raw = "Hello\x00World\x0b"
    result = clean_text(raw)
    assert "\x00" not in result
    assert "\x0b" not in result


def test_fixes_pdf_hyphenation_linebreaks():
    raw = "This is infor-\nmation extracted from a PDF."
    result = clean_text(raw)
    assert "information" in result
    assert "infor-\nmation" not in result


def test_empty_input_returns_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_strips_leading_trailing_whitespace():
    assert clean_text("   hello world   ") == "hello world"
