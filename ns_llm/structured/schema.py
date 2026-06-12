from pydantic import BaseModel


def convert_to_json_schema(schema: type[BaseModel] | dict) -> dict:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    raise TypeError(
        f"'schema' deve essere un dict o una sottoclasse di BaseModel, "
        f"non {type(schema).__name__}."
    )
