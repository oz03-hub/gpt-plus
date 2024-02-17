from load_dotenv import load_dotenv
import autogen

load_dotenv()


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    # filter_dict={
    #     "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    # },
)


def route(task_message: str):
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "cache_seed": 41,  # seed for caching and reproducibility
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
        },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "")
        .rstrip()
        .endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": True,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
        },
    )

    chat_res = user_proxy.initiate_chat(
        assistant,
        message=task_message,
        summary_method="reflection_with_llm",
        silent=True,
    )

    return chat_res
