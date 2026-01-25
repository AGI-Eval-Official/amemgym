"""Agentic Write & Internal Storage (AWI) assistant."""

from loguru import logger
from amemgym.utils import load_json, save_json, call_llm
from .prompts import IN_CONTEXT_MEMORY_UPDATE_PROMPT, get_in_context_hack_prompt
import os
import json
from datetime import datetime
from .base import BaseAgent


class InContextMemAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.in_context_memory = []
        self.local_msgs = []
        info_types = self.config.get("info_types", None)
        if info_types is None:
            self.memory_update_prompt = IN_CONTEXT_MEMORY_UPDATE_PROMPT
        else:
            logger.info(f"Using in-context memory update prompt with info types: {info_types}")
            self.memory_update_prompt = get_in_context_hack_prompt(info_types)
            logger.debug(f"memory_update_prompt: {self.memory_update_prompt}")

    def act(self, obs: str) -> str:
        new_msg = {"role": "user", "content": obs}
        sorted_memories = sorted(self.in_context_memory, key=lambda x: x["timestamp"])
        memories_str = "\n".join([f"- {entry['label']}: {entry['value']}" for entry in sorted_memories])
        system_prompt = f"You are a helpful AI. Respond according to memories of the user.\nUser memories ordered by time (earliest to latest):\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}] + self.local_msgs + [new_msg]
        response = call_llm(messages, self.config["llm_config"])
        new_response = {"role": "assistant", "content": response}
        self.add_msgs([new_msg, new_response])
        return response

    def load_state(self, local_dir: str):
        self.local_msgs = load_json(os.path.join(local_dir, "msg_history.json"))
        self.in_context_memory = load_json(os.path.join(local_dir, "in_context_memory.json"))

    def save_state(self, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        save_json(os.path.join(local_dir, "msg_history.json"), self.local_msgs)
        save_json(os.path.join(local_dir, "in_context_memory.json"), self.in_context_memory)

    def answer_question(self, question: str):
        new_msg = {"role": "user", "content": question}
        # system prompt w/ memories
        sorted_memories = sorted(self.in_context_memory, key=lambda x: x["timestamp"])
        memories_str = "\n".join([f"- {entry['label']}: {entry['value']}" for entry in sorted_memories])
        system_prompt = f"You are a helpful AI. Respond according to memories of the user.\nUser memories ordered by time (earliest to latest):\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}] + self.local_msgs + [new_msg]
        return call_llm(messages, self.config["llm_config"], return_token_usage=True)

    def add_msgs(self, messages: list):
        # load interactions to update internal state
        assert len(messages) == 2, "Only support two-turn interactions in one batch"
        limit = self.config["agent_config"]["update_bsz"] + \
            self.config["agent_config"]["local_length"]
        self.local_msgs += messages
        if len(self.local_msgs) >= limit:
            # update memory
            update_bsz = self.config["agent_config"]["update_bsz"]
            msgs_to_insert, self.local_msgs = self.local_msgs[:update_bsz], self.local_msgs[update_bsz:]
            logger.trace(
                f"Inserting {len(msgs_to_insert)} messages into memory.\n{[msg for msg in msgs_to_insert if msg['role'] == 'user']}")
            self._update_memory(msgs_to_insert)

    def _update_memory(self, messages: list):
        current_memories = {entry["label"]: entry["value"] for entry in self.in_context_memory}
        current_memories_str = json.dumps(current_memories, indent=2, ensure_ascii=False)
        conversation_str = json.dumps(messages, indent=2, ensure_ascii=False)
        memory_prompt = self.memory_update_prompt.format(
            current_memories=current_memories_str, conversation=conversation_str
        )
        memory_updates = call_llm(
            [{"role": "user", "content": memory_prompt}], self.config["llm_config"], json=True)
        memory_updates = json.loads(memory_updates)
        timestamp = datetime.now().strftime(
            '%Y-%m-%dT%H:%M:%S.%f')[:-3]  # ISO 8601 format
        logger.trace(f"memory update {timestamp}: {memory_updates}")

        # added new entries
        for entry in self.in_context_memory:
            if entry["label"] in memory_updates:
                entry["value"] = memory_updates[entry["label"]]
                entry["timestamp"] = timestamp
                del memory_updates[entry["label"]]

        # updated existing entries
        for label, value in memory_updates.items():
            assert label not in current_memories
            self.in_context_memory.append(
                {"label": label, "value": value, "timestamp": timestamp})

        logger.trace(f"updated memory {timestamp}: {self.in_context_memory}")

    def set_prompts(self, prompts):
        """Set memory update prompt"""
        if "memory_update_prompt" in prompts:
            prompt = prompts['memory_update_prompt']
            
            # Use a more sophisticated approach that only escapes single braces
            import re

            # First, protect our placeholder patterns
            prompt = prompt.replace('{current_memories}', '___PLACEHOLDER_CURRENT___')
            prompt = prompt.replace('{conversation}', '___PLACEHOLDER_CONVERSATION___')
            
            # Escape single braces that aren't already escaped
            # This regex finds single { or } that aren't part of {{ or }}
            prompt = re.sub(r'(?<!\{)\{(?!\{)', '{{', prompt)  # { not preceded by { and not followed by {
            prompt = re.sub(r'(?<!\})\}(?!\})', '}}', prompt)  # } not preceded by } and not followed by }
            
            # Restore placeholders
            prompt = prompt.replace('___PLACEHOLDER_CURRENT___', '{current_memories}')
            prompt = prompt.replace('___PLACEHOLDER_CONVERSATION___', '{conversation}')

            self.memory_update_prompt = prompt
