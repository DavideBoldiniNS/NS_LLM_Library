import pytest
from pydantic import BaseModel
from ns_llm.structured.schema import convert_to_json_schema


class Persona(BaseModel):
    name: str
    age: int


def test_pydantic_to_json_schema():
    result = convert_to_json_schema(Persona)
    assert isinstance(result, dict)
    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert result["properties"]["age"]["type"] == "integer"


def test_dict_passthrough():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = convert_to_json_schema(schema)
    assert result is schema


def test_invalid_type_raises_type_error():
    with pytest.raises(TypeError):
        convert_to_json_schema(42)
