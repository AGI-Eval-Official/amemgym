"""Evolvable agents that can update their prompts through self-reflection.

This module provides evolvable agent classes that extend base agent functionality
with the ability to evolve their prompts based on feedback from evaluation results.
"""

import os
import re
import json
from datetime import datetime
from loguru import logger

from amemgym.utils import call_llm, load_json, save_json, parse_json
from .awi import InContextMemAgent
from .mem0 import Mem0Agent, format_mem0_memories
from .prompts import (
    IN_CONTEXT_MEMORY_UPDATE_PROMPT,
    IN_CONTEXT_MEMORY_UPDATE_PROMPT_TEMPLATE,
    MEMORY_TYPES_SECTION,
    MINIMAL_MEMORY_PROMPT_V2,
    MINIMAL_FACT_EXTRACTION_PROMPT,
    MEDIUM_FACT_EXTRACTION_PROMPT,
)


class EvolvableInContextAgent(InContextMemAgent):
    """In-context memory agent with evolvable prompts.

    This agent extends InContextMemAgent with the ability to evolve its memory
    update prompt through self-reflection based on evaluation feedback.

    Attributes:
        evolution_config: Configuration dict for evolution behavior.
        evolution_history: List of evolution steps taken.
        memory_update_prompt: Current memory update prompt (can be evolved).
        memory_types_section: Memory types section (for info_type evolution).
    """

    def __init__(self, config):
        super().__init__(config)
        self.evolution_config = config.get('evolution_config', {})
        self.evolution_history = []

        # Store the memory update prompt as an instance variable instead of global constant
        self.init_prompts(
            init_prompt_type=self.evolution_config.get("init_prompt_type"))

    def init_prompts(self, init_prompt_type="minimal"):
        """Initialize different versions of the memory update prompt.

        Args:
            init_prompt_type: Type of initial prompt to use.
                - "minimal": Minimal V2 prompt
                - "info_type" or "guided_info_type": Template-based with memory types
                - "default": Standard full prompt
        """
        if init_prompt_type == "minimal":
            self.memory_update_prompt = MINIMAL_MEMORY_PROMPT_V2
            logger.debug("Using minimal memory update prompt V2")

        elif init_prompt_type in ["info_type", "guided_info_type"]:
            self.memory_types_section = MEMORY_TYPES_SECTION
            self.memory_update_prompt = IN_CONTEXT_MEMORY_UPDATE_PROMPT_TEMPLATE.format(
                memory_types_section=MEMORY_TYPES_SECTION)
            logger.debug("Using memory update prompt as default (info type update only)")

        elif init_prompt_type == "default":
            self.memory_update_prompt = IN_CONTEXT_MEMORY_UPDATE_PROMPT

        else:
            logger.debug("Using default memory update prompt")
            self.memory_update_prompt = IN_CONTEXT_MEMORY_UPDATE_PROMPT

    def get_current_prompts(self):
        """Get current memory update prompt.

        Returns:
            dict: Dictionary containing current prompts.
        """
        if self.evolution_config.get("init_prompt_type") in ["info_type", "guided_info_type"]:
            self.memory_update_prompt = IN_CONTEXT_MEMORY_UPDATE_PROMPT_TEMPLATE.format(
                memory_types_section=self.memory_types_section)

        return {
            "memory_update_prompt": self.memory_update_prompt
        }

    def set_prompts(self, prompts):
        """Set memory update prompt with proper brace escaping.

        Args:
            prompts: Dictionary containing prompts to set.
        """
        if "memory_update_prompt" in prompts:
            prompt = prompts['memory_update_prompt']

            # First, protect our placeholder patterns
            prompt = prompt.replace('{current_memories}', '___PLACEHOLDER_CURRENT___')
            prompt = prompt.replace('{conversation}', '___PLACEHOLDER_CONVERSATION___')

            # Escape single braces that aren't already escaped
            prompt = re.sub(r'(?<!\{)\{(?!\{)', '{{', prompt)
            prompt = re.sub(r'(?<!\})\}(?!\})', '}}', prompt)

            # Restore placeholders
            prompt = prompt.replace('___PLACEHOLDER_CURRENT___', '{current_memories}')
            prompt = prompt.replace('___PLACEHOLDER_CONVERSATION___', '{conversation}')

            self.memory_update_prompt = prompt

    def _update_memory(self, messages):
        """Update memory using instance variable prompt instead of global constant."""
        current_memories = {entry["label"]: entry["value"]
                            for entry in self.in_context_memory}
        current_memories_str = json.dumps(
            current_memories, indent=2, ensure_ascii=False)
        conversation_str = json.dumps(messages, indent=2, ensure_ascii=False)

        # Use instance variable instead of global IN_CONTEXT_MEMORY_UPDATE_PROMPT
        memory_prompt = self.memory_update_prompt.format(
            current_memories=current_memories_str, conversation=conversation_str
        )

        memory_updates = call_llm(
            [{"role": "user", "content": memory_prompt}], self.config["llm_config"], json=True)
        memory_updates = json.loads(memory_updates)
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        logger.trace(f"memory update {timestamp}: {memory_updates}")

        for entry in self.in_context_memory:
            if entry["label"] in memory_updates:
                entry["value"] = memory_updates[entry["label"]]
                entry["timestamp"] = timestamp
                del memory_updates[entry["label"]]

        for label, value in memory_updates.items():
            assert label not in current_memories
            self.in_context_memory.append(
                {"label": label, "value": value, "timestamp": timestamp})

        logger.trace(f"updated memory {timestamp}: {self.in_context_memory}")

    def answer_question(self, question):
        """Answer a question and return memories alongside the response.

        Modified to return memories alongside the answer for evolution feedback.

        Args:
            question: The question to answer.

        Returns:
            tuple: (memories_str, (response, usage))
        """
        new_msg = {"role": "user", "content": question}
        sorted_memories = sorted(
            self.in_context_memory, key=lambda x: x["timestamp"])
        memories_str = "\n".join(
            [f"- {entry['label']}: {entry['value']}" for entry in sorted_memories])
        system_prompt = f"You are a helpful AI. Respond according to memories of the user.\nUser memories ordered by time (earliest to latest):\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}
                    ] + self.local_msgs + [new_msg]

        # Return memories_str alongside the LLM response
        return memories_str, call_llm(messages, self.config["llm_config"], return_token_usage=True)

    def save_state(self, local_dir):
        """Extended save_state to include evolution state."""
        super().save_state(local_dir)

        evolution_path = os.path.join(local_dir, "evolution_state.json")
        evolution_data = {
            "current_prompts": self.get_current_prompts(),
            "evolution_history": self.evolution_history,
        }
        save_json(evolution_path, evolution_data)

    def load_state(self, local_dir):
        """Extended load_state that includes evolution state."""
        super().load_state(local_dir)

        evolution_path = os.path.join(local_dir, "evolution_state.json")
        if os.path.exists(evolution_path):
            evolution_data = load_json(evolution_path)
            self.set_prompts(evolution_data['current_prompts'])
            self.evolution_history = evolution_data['evolution_history']

    def _evolve_policy(self, feedback):
        """Use self-reflection to update memory update prompt.

        Args:
            feedback: Feedback data from evaluation results.

        Returns:
            dict: Contains new_prompts and changes made.
        """
        changes = {}
        current_prompts = self.get_current_prompts()
        new_prompts = {}

        for prompt_type, prompt in current_prompts.items():
            if prompt_type not in self.evolution_config["targets"]:
                continue

            if self.evolution_config.get("init_prompt_type") in ["info_type", "guided_info_type"]:
                # Only evolve info type part
                logger.debug("Evolving only memory info types section")
                guided = False
                if self.evolution_config.get("init_prompt_type") == "guided_info_type":
                    guided = True
                    logger.debug("Evolving with guided examples")

                messages = self._build_info_type_evolution_prompt(
                    self.memory_types_section, json.dumps(feedback, indent=2),
                    guided=guided)

                result = call_llm(messages, self.config["llm_config"])
                result = parse_json(result)

                # Get new memory_types_section
                new_types = result['new_types']
                new_changes = result['changes']

                # Set new prompt and memory_update_prompt
                self.memory_types_section = new_types
                new_prompt = IN_CONTEXT_MEMORY_UPDATE_PROMPT_TEMPLATE.format(
                    memory_types_section=self.memory_types_section)

            else:   # Evolve the whole prompt
                messages = self._build_evolution_prompt(
                    prompt_type, prompt, json.dumps(feedback, indent=2))

                result = call_llm(messages, self.config["llm_config"])
                result = parse_json(result)

                # Fixing input format errors
                new_prompt = result['new_prompt']
                new_changes = result['changes']

                if "{current_memories}" not in new_prompt:
                    new_prompt += "\n\nCurrent memories for the same user:\n{current_memories}"
                if "{conversation}" not in new_prompt:
                    new_prompt += "\n\nConversation:\n{conversation}"
                if all([kwd not in new_prompt for kwd in ["json", "JSON"]]):
                    new_prompt += "\n\nOutput only a JSON object with the new or updated information. If there is nothing to add, output {{}}. Your output will be used to update the current memories with a dict union operation in Python like `current_memories |= new_memory`. "

            new_prompts[prompt_type] = new_prompt
            changes[prompt_type] = new_changes

        if len(new_prompts) == 0:
            raise NotImplementedError(
                "No policy update (due to wrong parameter set 'targets' in self.evolution_config)")

        self.set_prompts(new_prompts)
        return {"new_prompts": new_prompts, "changes": changes}

    def _build_evolution_prompt(self, current_prompt_type: str, current_prompt: str, feedback_summary: str):
        """Build evolution prompt for memory update prompt improvement.

        Args:
            current_prompt_type: Type of the current prompt.
            current_prompt: Current prompt content.
            feedback_summary: JSON-formatted feedback summary.

        Returns:
            list: List of message dicts for LLM call.
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a senior prompt engineer. Improve a prompt used by an agent to extract and organize user memories from conversations:\n"
                "Constraints:\n"
                "- Only modify the instructions, examples, and information types to focus on.\n"
                "- Do NOT modify the parts specifying output formats (e.g. JSON format requirements).\n"
                "- Keep the {current_memories} and {conversation} placeholders intact.\n"
            )
        }

        user_msg = {
            "role": "user",
            "content": (
                "Current prompt type:\n"
                f"\n{current_prompt_type}\n\n"
                "Current prompt:\n"
                f"\n{current_prompt}\n\n"
                "Feedback summary (from recent usage and preferences):\n"
                f"{feedback_summary}\n\n"
                "Task:\n"
                "- Propose improved memory extraction prompt reflecting the feedback.\n"
                "- Only modify instructions, examples, and information types.\n"
                "- Keep JSON format requirements and placeholders unchanged.\n"
                "Output JSON schema (return ONLY this JSON):\n"
                "```json {\n"
                '  "new_prompt": "string",\n'
                '  "changes": ["short bullet of what changed", "..."]\n'
                "} ```"
            )
        }
        return [system_msg, user_msg]

    def _build_info_type_evolution_prompt(self, current_memory_types_section: str, feedback_summary: str, guided: bool):
        """Build evolution prompt for memory update prompt improvement (memory types section ONLY).

        Args:
            current_memory_types_section: Current memory types section content.
            feedback_summary: JSON-formatted feedback summary.
            guided: Whether to include guided examples.

        Returns:
            list: List of message dicts for LLM call.
        """
        query_schema_demonstrations = ""
        if guided:
            # Add question, feedback -> schema examples
            query_schema_key_pairs = [
                ("How can I plan a solo backpacking trip to Coronado National Forest that's safe and enjoyable?",
                 [
                    "solo_backpacking_experience_level",
                    "solo_backpacking_trip_duration"
                    ]),
                ("What are some engaging non-fiction books on history or science that align with my interests?",
                 [
                    "non_fiction_reading_goal_type",
                    "preferred_book_format"
                    ]),
                ("What are the best ways to balance work and personal interests like hiking and woodcarving?",
                 [
                    "work_schedule_pattern",
                    "personal_activity_energy_level"
                    ]),
                ("Which strategies can I use to motivate my retail team when facing a sales slump?",
                 [
                    "retail_team_size_category",
                    "retail_sales_trend_last_quarter"
                    ])
            ]
            query_schema_demonstrations = "\nHere are example memory types that should be inferred from observed questions in feedback:\n\n" + "\n".join([
                    f"Observed questions: {q}\nMemory types: {', '.join(keys)}\n"
                    for q, keys in query_schema_key_pairs
                ])

        # NOTE: Match mem-env behavior - when guided=False, system content becomes ""
        # This is due to Python operator precedence: (str + str) if guided else ""
        system_msg = {
            "role": "system",
            "content": (
                "You are a senior prompt engineer. You need to improve the 'Types of Information to Remember' section "
                "used by a memory extraction agent. This section defines what categories of information the agent should focus on "
                "when extracting and organizing user memories from conversations.\n\n"
                "Constraints:\n"
                "- Focus on making the types more specific and actionable based on feedback.\n"
                "- Each type should be clear about what information to extract and store.\n"
            ) + query_schema_demonstrations if guided else ""
        }

        user_msg = {
            "role": "user",
            "content": (
                "Current 'Types of Information to Remember' section:\n\n"
                f"{current_memory_types_section}\n\n"
                "Feedback summary (from recent usage and evaluation):\n"
                f"{feedback_summary}\n\n"
                "Task:\n"
                "- Improve the types of information to remember based on the feedback.\n"
                "- Keep a similar format with clear descriptions.\n"
                "Output JSON schema (return ONLY this JSON):\n"
                "```json {\n"
                '  "new_types": "string (the improved types section)",\n'
                '  "changes": ["short bullet of what changed", "..."]\n'
                "} ```"
            )
        }
        return [system_msg, user_msg]


class EvolvableMem0Agent(Mem0Agent):
    """Mem0 agent with evolvable fact extraction prompts.

    This agent extends Mem0Agent with the ability to evolve its fact
    extraction prompt through self-reflection based on evaluation feedback.

    Attributes:
        evolution_config: Configuration dict for evolution behavior.
        evolution_history: List of evolution steps taken.
    """

    def __init__(self, config):
        super().__init__(config)
        self.evolution_config = config.get('evolution_config', {})
        self.evolution_history = []

        self.init_default_prompts(
            init_prompt_type=self.evolution_config.get("init_prompt_type"))

    def init_default_prompts(self, init_prompt_type="minimal"):
        """Initialize different versions of the fact extraction prompt.

        Args:
            init_prompt_type: Type of initial prompt to use.
                - "minimal": Minimal fact extraction prompt
                - "medium": Medium fact extraction prompt with types
                - default: Use Mem0's default heavy prompt
        """
        if init_prompt_type == "minimal":
            self.set_prompts(
                {"fact_extraction_prompt": MINIMAL_FACT_EXTRACTION_PROMPT})
            logger.debug(
                "Using minimal memory extraction prompt. (Minimal instruction)")
        elif init_prompt_type == "medium":
            self.set_prompts(
                {"fact_extraction_prompt": MEDIUM_FACT_EXTRACTION_PROMPT})
            logger.debug(
                "Using medium memory extraction prompt. (Medium instruction)")
        else:
            logger.debug(
                "Using default memory extraction prompt. (Heavy instruction)")

    def get_current_prompts(self):
        """Get current extraction and update prompts.

        Returns:
            dict: Dictionary containing current prompts.
        """
        import mem0.configs.prompts
        return {
            "fact_extraction_prompt": self.memory.config.custom_fact_extraction_prompt or mem0.configs.prompts.FACT_RETRIEVAL_PROMPT,
            "update_memory_prompt": self.memory.config.custom_update_memory_prompt or mem0.configs.prompts.DEFAULT_UPDATE_MEMORY_PROMPT
        }

    def set_prompts(self, prompts):
        """Set extraction and update prompts.

        Args:
            prompts: Dictionary containing prompts to set.
        """
        if "fact_extraction_prompt" in prompts:
            self.memory.config.custom_fact_extraction_prompt = prompts['fact_extraction_prompt']
        if "update_memory_prompt" in prompts:
            self.memory.config.custom_update_memory_prompt = prompts['update_memory_prompt']

    def answer_question(self, question):
        """Answer a question and return memories alongside the response.

        Modified to return memories alongside the answer for evolution feedback.

        Args:
            question: The question to answer.

        Returns:
            tuple: (memories_str, (response, usage))
        """
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
        messages = [{"role": "system", "content": system_prompt}
                    ] + self.local_msgs + [new_msg]

        return memories_str, call_llm(messages, self.config["llm_config"], return_token_usage=True)

    def save_state(self, local_dir):
        """Extended save_state to include evolution state.

        Args:
            local_dir: Directory to save state to.
        """
        super().save_state(local_dir)

        evolution_path = os.path.join(local_dir, "evolution_state.json")
        evolution_data = {
            "current_prompts": self.get_current_prompts(),
            "evolution_history": self.evolution_history,
        }
        save_json(evolution_path, evolution_data)

    def load_state(self, local_dir):
        """Extended load_state that includes evolution state.

        Args:
            local_dir: Directory to load state from.
        """
        super().load_state(local_dir)

        evolution_path = os.path.join(local_dir, "evolution_state.json")
        if os.path.exists(evolution_path):
            evolution_data = load_json(evolution_path)
            self.set_prompts(evolution_data['current_prompts'])
            self.evolution_history = evolution_data['evolution_history']

    def _evolve_policy(self, feedback):
        """Use self-reflection to update prompts.

        Args:
            feedback: Feedback data from evaluation results.

        Returns:
            dict: Contains new_prompts and changes made.
        """
        changes = {}
        current_prompts = self.get_current_prompts()
        new_prompts = {}

        for prompt_type, prompt in current_prompts.items():
            if prompt_type not in self.evolution_config["targets"]:
                continue
            messages = self._build_evolution_prompt(
                prompt_type, prompt, json.dumps(feedback, indent=2))

            result = call_llm(messages, self.config["llm_config"])
            result = parse_json(result)

            # Fixing input format errors
            new_prompt = result['new_prompt']

            if "{current_memories}" not in new_prompt:
                new_prompt += "\n\nCurrent memories for the same user:\n{current_memories}"
            if "{conversation}" not in new_prompt:
                new_prompt += "\n\nConversation:\n{conversation}"
            if all([kwd not in new_prompt for kwd in ["json", "JSON"]]):
                new_prompt += "\n\nOutput only a JSON object with the new or updated information. If there is nothing to add, output {{}}. Your output will be used to update the current memories with a dict union operation in Python like `current_memories |= new_memory`. "

            new_prompts[prompt_type] = new_prompt
            changes[prompt_type] = result['changes']

        if len(new_prompts) == 0:
            raise NotImplementedError(
                "No policy update (due to wrong parameter set 'targets' in self.evolution_config)")

        self.set_prompts(new_prompts)
        return {"new_prompts": new_prompts, "changes": changes}

    def _build_evolution_prompt(
        self,
        current_prompt_type: str,
        current_prompt: str,
        feedback_summary: str
    ):
        """Build evolution prompt for prompt improvement.

        Args:
            current_prompt_type: Type of the current prompt.
            current_prompt: Current prompt content.
            feedback_summary: JSON-formatted feedback summary.

        Returns:
            list: List of message dicts for LLM call.
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a senior prompt engineer. Improve a prompt used by an agent assistant to create or update its own memories based on conversations with the user:\n"
                "Constraints:\n"
                "- Only modify the few shot examples and types of information to remember.\n"
                "- Do NOT modify the parts specifying output formats (e.g. JSON).\n"
            )
        }

        user_msg = {
            "role": "user",
            "content": (
                "Current prompt type:\n"
                f"\n{current_prompt_type}\n\n"
                "Current prompt:\n"
                f"\n{current_prompt}\n\n"
                "Feedback summary (from recent usage and preferences):\n"
                f"{feedback_summary}\n\n"
                "Task:\n"
                "- Propose improved prompts reflecting the feedback.\n"
                "- Only modify the few shot examples and types of information to remember.\n"
                "- Do NOT modify the parts specifying output formats (e.g. JSON).\n"
                "Output JSON schema (return ONLY this JSON):\n"
                "```json {\n"
                '  "new_prompt": "string",\n'
                '  "changes": ["short bullet of what changed", "..."]\n'
                "} ```"
            )
        }
        return [system_msg, user_msg]
