import subprocess
import re
import json
import tempfile
import os
from enum import Enum

# from utils import call_chatgpt_analyze_static_security


class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets


def extract_function_name(code):
    # Extract the function name from the code using regex
    match = re.search(r'def (\w+)\s*\(', code)
    if match:
        return match.group(1)
    raise ValueError("Function name could not be extracted from the code.")


def remove_json_prefix(input_str):
    # Check if the input string starts with "json\n"
    if input_str.startswith("json\n"):
        # Remove the prefix and return the remaining string
        return input_str[len("json\n"):]
    return input_str


class ExecutorStaticAgent:
    def __init__(self, entry):
        self.entry = entry
        
    def execute_static_analysis_gpt(self, code, codeql_agent, bandit_agent):
        codeql_static_output = codeql_agent.analyze_code(code)
        bandit_static_output = bandit_agent.analyze_code(code)
        if len(codeql_static_output) == 0 and len(bandit_static_output) == 0:
            return FResult.SAFE, "No Issues Found"
        else:
            return FResult.ERROR, (codeql_static_output, bandit_static_output)
