import time
from openai import OpenAI
from loguru import logger
from backoff import on_exception, expo


@on_exception(expo, Exception, max_tries=10)
def call_llm(messages, llm_config, json=False, return_token_usage=False):
    if not hasattr(call_llm, "clients"):
        call_llm.clients = {}
    start_time = time.time()
    try:
        if llm_config["base_url"] not in call_llm.clients:
            call_llm.clients[llm_config["base_url"]] = OpenAI(
                base_url=llm_config["base_url"],
                api_key=llm_config["api_key"]
            )
        client = call_llm.clients[llm_config["base_url"]]
        extra_params = llm_config.get("extra_params", {})
        response = client.chat.completions.create(
            model=llm_config["llm_model"],
            messages=messages,
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            response_format={"type": "json_object"} if json else None,
            **extra_params
        )
        time_elapsed = time.time() - start_time
        logger.trace(
            f"Call llm finished. Model: {llm_config['llm_model']}, "
            f"Time elapsed: {time_elapsed} seconds. "
            f"Number of tokens (prompt/completion/total): {response.usage.prompt_tokens}/"
            f"{response.usage.completion_tokens}/{response.usage.total_tokens}. "
            f"Source: {llm_config.get('source', 'unknown')}"
        )
        logger.trace("Prompt:\n" + messages[-1]["content"])
        logger.trace("Response:\n" + response.choices[0].message.content)
        if return_token_usage:
            return response.choices[0].message.content, {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.total_tokens, "time_elapsed": time_elapsed}
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Call LLM Failed: {e}")
        raise e
