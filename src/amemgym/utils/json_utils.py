import re
import json


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.loads(f.read())
    

def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_json(response):
    """
    Parse JSON content from agent responses, handling various formatting issues.
    
    This function extracts and parses JSON from agent responses that may contain
    additional text or be wrapped in markdown code blocks. It's designed to be
    robust against common formatting variations in LLM outputs.
    
    Args:
        response (str): Raw response text from the agent
    
    Returns:
        dict or None: Parsed JSON object, or None if parsing fails
    """
    # remove ```json``` and ``` from the response
    json_part = re.search(r'```json(.*?)```', response, re.DOTALL)
    if json_part:
        response = json_part.group(1).strip()
    return json.loads(response)
