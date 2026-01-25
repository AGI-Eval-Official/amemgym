"""Base agent for assistants."""

class BaseAgent:
    """Base agent for necessary interface."""

    def reset(self):
        """reset internal state for new episode"""
        raise NotImplementedError

    def act(self, obs: str) -> str:
        """generate response based on the observation (user input) and modify internal state"""
        raise NotImplementedError

    def add_msgs(self, msgs: list):
        """modify internal state based on additional messages"""
        raise NotImplementedError

    def load_state(self, local_dir: str):
        """read internal state from a specific directory"""
        raise NotImplementedError

    def save_state(self, local_dir: str):
        """write internal state to a specific directory"""
        raise NotImplementedError

    def answer_question(self, question: str):
        """respond to a question based on internal state (without modifying internal state)"""
        raise NotImplementedError
