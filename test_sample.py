import pytest
from ai import AI

ai = AI()
ai.handle_message("Hello", "text")

print(ai.contexts[ai.current_context])
print(ai.contexts)

print(ai.handle_message("A dog playing in the water", "image"))


def test_current_context():
    ai = AI()
    assert ai.current_context == None

    ai.handle_message("Hello", "text")
    assert ai.current_context == "0"

    ai.handle_message("Hello Again", "text")
    assert ai.current_context == "0"

    # SWITCH CONTEXT
    ai.switch_context("10")
    assert ai.current_context == "10"

    ai.handle_message("New Hello", "text")
    assert ai.current_context == "10"


def test_non_empty_messages():
    ai = AI()
    assert ai.current_context == None

    ai.handle_message("Hello", "text")
    assert len(ai.contexts[ai.current_context]) == 3

    ai.switch_context("10")
    ai.handle_message("New Hello", "text")
    assert len(ai.contexts[ai.current_context]) == 3

    ai.switch_context("0")  # revert
    ai.handle_message("Hello Again", "text")
    assert len(ai.contexts[ai.current_context]) == 5


def test_text_2_text_route():
    ai = AI()

    try:
        r = ai.handle_message("Hello", "text")
    except Exception as e:
        pytest.fail("Did not expect error: {}".format(e))


def test_text_2_image_route():
    ai = AI()

    try:
        r = ai.handle_message("Dog playing in water", "image")
    except Exception as e:
        pytest.fail("Did not expect error: {}".format(e))
