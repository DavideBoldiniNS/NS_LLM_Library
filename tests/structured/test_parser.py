import pytest
from ns_llm.structured.parser import parse_json_response


def test_direct_json():
    text = '{"name": "Marco", "age": 30}'
    result = parse_json_response(text)
    assert result == {"name": "Marco", "age": 30}


def test_markdown_json_block():
    text = "Ecco il risultato:\n```json\n{\"name\": \"Marco\"}\n```"
    result = parse_json_response(text)
    assert result == {"name": "Marco"}


def test_no_json_raises_value_error():
    with pytest.raises(ValueError, match="Impossibile parsare"):
        parse_json_response("Ciao, come stai?")


def test_empty_string_raises():
    with pytest.raises(ValueError, match="Impossibile parsare"):
        parse_json_response("")


def test_markdown_without_json_tag_not_parsed():
    text = "```\n{'name': 'test'}\n```"
    with pytest.raises(ValueError, match="Impossibile parsare"):
        parse_json_response(text)
