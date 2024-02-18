import curses
from ai import AI
import time

# Set up variables and objects
ai_client = AI()


def parse_command(input_string: str) -> str:
    if not input_string.startswith("!"):
        return handle_inference_command(input_string)

    # Split the input into command and arguments
    command, *arguments = input_string.split()
    body = " ".join(arguments)

    # Check the command and handle accordingly
    if command == "!text":
        return handle_text_command(body)
    elif command == "!image":
        return handle_image_command(body)
    elif command == "!autogen":
        return handle_autogen_command(body)
    elif command == "!switch":
        return handle_switch_command(body)
    else:
        return "Unknown command"


def handle_text_command(body: str) -> str:
    # Implement the logic for handling !text command
    if body:
        response = ai_client.handle_message(body, "text")
        return response
    else:
        return "Invalid !text command"


def handle_image_command(body: str) -> str:
    # Implement the logic for handling !image command
    if body:
        response_url = ai_client.handle_message(body, "image")
        return response_url
    else:
        return "Invalid !image command"


def handle_autogen_command(body: str) -> str:
    # Implement the logic for handling !autogen command
    if body:
        chat_res = ai_client.handle_message(body, "autogen")
        return f"""SUMMARY: {chat_res.summary}\nACCESS FULL CHAT: !switch {ai_client.current_context}-autogen"""
    else:
        return "Invalid !autogen command"


def handle_switch_command(body: str) -> list[dict[str | str]]:
    # Implement the logic for handling !switch command
    if body:
        messages = ai_client.switch_context(body)
        return messages
    else:
        return "Invalid !switch command"


def handle_list_command(_body: str) -> str:
    return ai_client.list_context()


def handle_inference_command(body: str) -> str:
    if body:
        response = ai_client.handle_message(body, "infer")
        return response
    else:
        return "Invalid !inference command"


def main(stdscr):

    # Set up the screen
    curses.curs_set(0)
    stdscr.clear()
    stdscr.refresh()

    # Initialize variables
    messages = []
    input_buffer = ""

    # Main loop
    while True:
        # Display messages
        stdscr.clear()
        for i, message in enumerate(messages):
            stdscr.addstr(i, 0, message)

        # Display user input
        stdscr.addstr(len(messages) + 10, 0, "> " + input_buffer)

        # Refresh the screen
        stdscr.refresh()

        # Get user input
        key = stdscr.getch()

        # Process user input
        if key == 10:  # Enter key
            if input_buffer:
                messages.append("You: " + input_buffer)

                # AI response
                # Pre-process
                ai_response = parse_command(input_buffer)

                messages.append(ai_response)
                input_buffer = ""
        elif key == 127:  # Backspace key
            input_buffer = input_buffer[:-1]
        elif key >= 32 and key <= 126:  # Printable ASCII characters
            input_buffer += chr(key)


def display_large_string(stdscr, text):
    max_y, max_x = stdscr.getmaxyx()
    y, x = stdscr.getyx()
    lines = text.splitlines()

    for line in lines:
        if y < max_y - 1:
            stdscr.addstr(y, x, line[: max_x - 1])
            y += 1
        else:
            stdscr.addstr(max_y - 1, x, "...")
            break


if __name__ == "__main__":
    curses.wrapper(main)
