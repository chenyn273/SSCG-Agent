import re


def preprocess_string(input_string, lg):
    if f"```{lg}\n" in input_string:
        input_string = input_string[input_string.find(
            f"```{lg}\n") + len(f"```{lg}\n"):]
        input_string = input_string[:input_string.find("```")-1]
    elif "```" in input_string:
        input_string = input_string[input_string.find("```") + 4:]
        input_string = input_string[:input_string.find("```")-1]

    return input_string


class ProgrammerAgent:
    def __init__(self, entry, llm):
        self.entry = entry
        self.llm = llm

    def call_chatgpt_programmer(self, prompt):
        text = f"""Please generate the complete and runnable Python code based on the following code snippet.
Complete any missing implementation details, but do not generate any test code.
Output the completed code wrapped in ```python```.
{prompt}
"""
        try:
            completion = self.llm.generate(text)
            completion = completion.strip()
            completion = preprocess_string(completion, "python")

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def call_chatgpt_programmer_feedback_static(self, completion, analyze_res):
        gpt_prompt = f'''Please analyze the original code provided below:
```python
{completion}
```
Based on the issues and recommended fixes described below:
{analyze_res}
# Follow the steps below:
Step 1. Think step by step in plain text (no code blocks), explain the reason for the issues.  
Step 2. Reason about how to fix it.  
Step 3. At the end, output the final fixed code wrapped in ```python```.
'''
        try:
            completion = self.llm.generate(gpt_prompt)
            completion = completion.strip()
            pattern = r"```python\s*(.*?)```"
            matches = re.findall(pattern, completion, re.DOTALL)
            completion = matches[-1]

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def call_chatgpt_programmer_feedback_fuzzing(self, completion, fuzzing_feedback):

        gpt_prompt = f'''Here is the original Python code:
```python
{completion}
```
The fuzzing test result analysis: {fuzzing_feedback}
# Follow the steps below:
Step 1. Think step by step in plain text (no code blocks), explain the reason for the failure.  
Step 2. Reason about how to fix it.  
Step 3. At the end, output the final fixed code wrapped in ```python```.  
'''
        try:
            completion = self.llm.generate(gpt_prompt)
            completion = completion.strip()
            pattern = r"```python\s*(.*?)```"
            matches = re.findall(pattern, completion, re.DOTALL)
            completion = matches[-1]

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def call_chatgpt_programmer_feedback_func(self, completion, inputs):
        gpt_prompt = f'''Here is the original Python code:
```python
{completion}
```
The unit test result shows the following problem: {inputs}
# Follow the steps below:
Step 1. Think step by step in plain text (no code blocks), explain the reason for the failure.  
Step 2. Reason about how to fix it.  
Step 3. At the end, output the final corrected code wrapped in ```python```.  
'''
        try:
            completion = self.llm.generate(gpt_prompt)
            completion = completion.strip()
            pattern = r"```python\s*(.*?)```"
            matches = re.findall(pattern, completion, re.DOTALL)
            completion = matches[-1]

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def write_code(self,):
        prompt = self.entry['Prompt']
        code = self.call_chatgpt_programmer(prompt)
        return code

    def write_code_feedback_static(self, completion, cwe_code):
        code = self.call_chatgpt_programmer_feedback_static(
            completion, cwe_code)
        return code

    def write_code_feedback_fuzz(self, completion, fuzzing_feedback):
        code = self.call_chatgpt_programmer_feedback_fuzzing(
            completion, fuzzing_feedback)
        return code

    def write_code_feedback_func(self, completion, inputs):
        code = self.call_chatgpt_programmer_feedback_func(
            completion, inputs)
        return code
