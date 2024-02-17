from load_dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
import tiktoken
import os

from autogenmod import route

load_dotenv()

TOKEN_LIMIT = int(os.environ["TOKEN_LIMIT"])


def num_tokens_from_messages(messages, model=os.environ["TEXT_MODEL"]):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class AI:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.text_model_name = os.environ["TEXT_MODEL"]
        self.image_model_name = os.environ["IMAGE_MODEL"]
        self.contexts = defaultdict(
            lambda: [{"role": "system", "content": "You are a helpful AI assistant."}]
        )
        self.current_context = None

    def switch_context(self, switch_to: str):
        self.current_context = switch_to
        return self.contexts[self.current_context]

    def handle_text_to_text_message(self, new_message: str):
        context_messages = self.contexts[self.current_context]

        context_messages.append({"role": "user", "content": new_message})

        if num_tokens_from_messages(context_messages) > TOKEN_LIMIT:
            context_messages = context_messages[-5:]  # trim quickly

        response = self.text_to_text(context_messages)
        context_messages.append({"role": "assistant", "content": response})
        return response

    def handle_text_to_image(self, new_message: str):
        return self.text_to_image(new_message)

    def handle_autogen(self, new_message: str):
        chat_res = route(new_message)
        new_context = self.current_context + "-autogen"
        self.contexts[new_context] = chat_res.chat_history

        return chat_res.summary + " ### AT {}".format(new_context)

    def handle_message(self, new_message: str, command: str):
        if self.current_context == None:
            self.current_context = "0"

        if command == "text":
            return self.handle_text_to_text_message(new_message)  # str
        elif command == "image":
            return self.handle_text_to_image(new_message)  # url str
        elif command == "autogen":
            return self.handle_autogen(new_message)
        else:
            pass  # bayes inference route

    def text_to_text(self, messages: list[dict[str | str]]):
        response = self.client.chat.completions.create(
            messages=messages, model=self.text_model_name
        )
        return response.choices[0].message.content

    def text_to_image(self, prompt: str):
        response = self.client.images.generate(
            model=self.image_model_name,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
