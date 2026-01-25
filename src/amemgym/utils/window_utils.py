import tiktoken
from loguru import logger


def count_tokens(messages: list[dict], model: str = "text-embedding-3-small", scaling_factor=1.0, extra_per_msg=4) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError as e:
        logger.warning(e)
        encoding = tiktoken.get_encoding("o200k_base")
        logger.warning("using tiktoken o200k_base for estimation")
    total_tokens = sum(len(encoding.encode(m.get("content", ""))) for m in messages) + len(messages) * extra_per_msg
    logger.trace(f"counted {total_tokens} tokens for {len(messages)} msgs")
    return int(total_tokens * scaling_factor)
