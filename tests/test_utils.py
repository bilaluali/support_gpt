from unittest.mock import patch

from chat.utils import CollectedData, OpenAIResponse, parse_response


def test_parse_response():
    collected_data = CollectedData(
        order_number="1234567890",
        problem_category="technical",
        problem_description="The product is not working",
        urgency_level="high",
    )
    response_content = (
        "All good, how can I help you today?"
        + "<COLLECTED_DATA>"
        + collected_data.model_dump_json()
        + "</COLLECTED_DATA>"
    )
    response = parse_response(response_content)
    assert isinstance(response, OpenAIResponse)
    assert response.reply == "All good, how can I help you today?"
    assert response.collected_data == collected_data


@patch("chat.utils.logger")
def test_parse_response_no_collected_data_tag(m_logger):
    response_content = "All good, how can I help you today?"
    response = parse_response(response_content)
    assert isinstance(response, OpenAIResponse)
    assert response.reply == "All good, how can I help you today?"
    assert response.collected_data == CollectedData()
    m_logger.warning.assert_called_once_with(
        "No <COLLECTED_DATA> block found in response."
    )


@patch("chat.utils.logger")
def test_parse_response_invalid_collected_data_json(m_logger):
    response_content = "All good, how can I help you today? <COLLECTED_DATA>invalid json</COLLECTED_DATA>"
    response = parse_response(response_content)
    assert isinstance(response, OpenAIResponse)
    assert response.reply == "All good, how can I help you today?"
    assert response.collected_data == CollectedData()
    assert (
        "Invalid JSON in <COLLECTED_DATA>: invalid json"
        in m_logger.error.call_args[0][0]
    )
