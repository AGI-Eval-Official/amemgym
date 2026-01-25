from amemgym.utils import call_llm
from loguru import logger
import os
import json
import shutil
from copy import deepcopy
from backoff import on_exception, expo
from .base import BaseAgent


@on_exception(expo, Exception, max_tries=10)  # retry on failure
def insert_mem0(memory, batch, user_id, infer):
    """Insert a batch of messages into memory."""
    memory_log = memory.add(batch, user_id=user_id, infer=infer)
    logger.trace(f"Memory log: {memory_log}")


def format_mem0_memories(memories):
    def get_time(memory):
        if memory["updated_at"]:
            return memory["updated_at"][:19]
        return memory["created_at"][:19]
    sorted_memories = sorted(memories["results"], key=get_time)
    memories_str = "\n".join([
        f"- {entry['memory']}"
        for entry in sorted_memories
    ])
    return memories_str


class Mem0Agent(BaseAgent):
    def __init__(self, config):
        from mem0 import Memory
        self.config = deepcopy(config)
        mem_dir = self.config.get("local_mem_dir")
        if os.path.exists(mem_dir):
            shutil.rmtree(mem_dir)
        os.makedirs(mem_dir, exist_ok=True)
        self.config["memory_config"]["vector_store"]["config"]["url"] = os.path.join(
            mem_dir, "mem.db")
        if "graph_store" in self.config["memory_config"]:
            self.config["memory_config"]["graph_store"]["config"]["db"] = os.path.join(mem_dir, "mem-graph.kuzu")
        self.config["memory_config"]["history_db_path"] = os.path.join(
            mem_dir, "mem_hist.db")
        self.memory = Memory.from_config(self.config["memory_config"])
        self.reset()

    def reset(self):
        self.memory.reset()
        self.local_msgs = []

    def act(self, obs: str) -> str:
        new_msg = {"role": "user", "content": obs}
        # retrieve
        relevant_memories = self.memory.search(
            query=obs,
            user_id="USER",
            limit=self.config["agent_config"]["top_k"]
        )

        # system prompt w/ memories
        memories_str = format_mem0_memories(relevant_memories)
        logger.trace(f"Retrieved memories: {memories_str}")
        system_prompt = f"You are a helpful AI. Respond according to retrieved memories.\nRelevant user memories ordered by time (earliest to latest):\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}] + self.local_msgs + [new_msg]
        response = call_llm(messages, self.config["llm_config"])
        new_response = {"role": "assistant", "content": response}

        # record current interaction
        self.add_msgs(messages=[new_msg, new_response])

        return response

    def load_state(self, local_dir: str):
        from mem0 import Memory
        del self.memory
        shutil.rmtree(self.config["local_mem_dir"])
        shutil.copytree(local_dir, self.config["local_mem_dir"])
        self.memory = Memory.from_config(self.config["memory_config"])
        with open(os.path.join(local_dir, "msg_history.json"), "r") as f:
            self.local_msgs = json.load(f)

    def save_state(self, local_dir: str):
        shutil.copytree(self.config["local_mem_dir"], local_dir)
        with open(os.path.join(local_dir, "msg_history.json"), "w") as f:
            json.dump(self.local_msgs, f, indent=2, ensure_ascii=False)

    def answer_question(self, question: str):
        new_msg = {"role": "user", "content": question}
        # retrieve
        relevant_memories = self.memory.search(
            query=question,
            user_id="USER",
            limit=self.config["agent_config"]["top_k"]
        )
        # system prompt w/ memories
        memories_str = format_mem0_memories(relevant_memories)
        logger.trace(f"Retrieved memories: {memories_str}")
        system_prompt = f"You are a helpful AI. Respond according to retrieved memories.\nRelevant user memories ordered by time (earliest to latest):\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}] + self.local_msgs + [new_msg]
        return call_llm(messages, self.config["llm_config"], return_token_usage=True)

    def add_msgs(self, messages: list):
        """Add messages."""
        assert len(messages) == 2, "Only support two-turn interactions in one batch"
        limit = self.config["agent_config"]["update_bsz"] + \
            self.config["agent_config"]["local_length"]
        infer = self.config["agent_config"]["enable_llm_mem_policy"]
        self.local_msgs += messages
        if len(self.local_msgs) >= limit:
            # update memory
            update_bsz = self.config["agent_config"]["update_bsz"]
            msgs_to_insert, self.local_msgs = self.local_msgs[:update_bsz], self.local_msgs[update_bsz:]
            logger.trace(
                f"Inserting {len(msgs_to_insert)} messages into memory.\n{[msg for msg in msgs_to_insert if msg['role'] == 'user']}")
            for msg in msgs_to_insert:
                if msg["role"] == "user":
                    msg["content"] = f"USER INPUT: " + msg["content"]
                elif msg["role"] == "assistant":
                    msg["content"] = f"ASSISTANT RESPONSE: " + msg["content"]
                else:
                    raise ValueError(f"Unknown message role: {msg['role']}")
            insert_mem0(self.memory, msgs_to_insert, user_id="USER", infer=infer)
