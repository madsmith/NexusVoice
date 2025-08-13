from typing import List
from nexusvoice.internal.api.base import ModelResponse
from pydantic_ai.messages import TextPart, ModelResponsePart


def test_modelresponse_text_property():
    # Create a ModelResponse with multiple parts
    parts: List[ModelResponsePart] = [
        TextPart(content="Hello, "),
        TextPart(content="world!"),
        TextPart(content=" Goodbye.")
    ]
    response = ModelResponse(parts=parts)
    assert response.text == "Hello, world! Goodbye."


def test_modelresponse_empty_parts():
    response = ModelResponse(parts=[])
    assert response.text == ""


def test_modelresponse_non_text_parts():
    class DummyPart:
        pass
    response = ModelResponse(parts=[DummyPart()]) # type: ignore
    # Should not raise, should just skip parts without 'content'
    assert response.text == ""
