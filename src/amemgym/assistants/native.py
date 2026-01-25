from amemgym.utils import load_json, save_json, call_llm
from .base import BaseAgent
import os


class NaiveAgent(BaseAgent):
    """Naive agent wrapper for LLMs interacting with AMemGym."""

    def __init__(self, config):
        self.llm_config = config
        self.reset()

    def reset(self):
        self.msg_history = []

    def act(self, obs: str) -> str:
        self.msg_history.append({"role": "user", "content": obs})
        response = call_llm(self.msg_history, self.llm_config)
        self.msg_history.append({"role": "assistant", "content": response})
        return response

    def add_msgs(self, msgs: list):
        self.msg_history.extend(msgs)

    def load_state(self, local_dir: str):
        self.msg_history = load_json(os.path.join(local_dir, "msg_history.json"))

    def save_state(self, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        save_json(os.path.join(local_dir, "msg_history.json"), self.msg_history)

    def answer_question(self, question: str):
        msg = {"role": "user", "content": question}
        return call_llm(self.msg_history + [msg], self.llm_config, return_token_usage=True)
