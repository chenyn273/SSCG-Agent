from copy import deepcopy
import random
import string
import json
# from executor_agent_safe import remove_json_prefix


def preprocess_string(input_string, lg):
    if f"```{lg}\n" in input_string:
        input_string = input_string[input_string.find(
            f"```{lg}\n") + len(f"```{lg}\n"):]
        input_string = input_string[:input_string.find("```")-1]
    elif "```" in input_string:
        input_string = input_string[input_string.find("```") + 4:]
        input_string = input_string[:input_string.find("```")-1]

    return input_string


class TesterFuzzAgent:
    def __init__(self, entry, llm):
        self.entry = entry
        self.llm = llm

        prompt_path_fuzz = "./prompts_fewshot/initial_inputs_prompt.txt"
        with open(prompt_path_fuzz, "r") as f:
            self.construct_few_shot_prompt_fuzz = f.read()

    def call_chatgpt_fuzzing_tester(self, prompt):
        text = f"""{self.construct_few_shot_prompt_fuzz}

## Prompt 2:
```python
{prompt}
```
## Completion 2:
"""
        try:
            new_completion = self.llm.generate(
                "You are a code tester specialized in fuzzing.\n" + text)
            new_completion = new_completion.strip()
            new_completion = preprocess_string(new_completion, "json")
        except Exception as e:
            print(e)
            new_completion = ""

        return new_completion

    def generate_test_inputs(self):
        prompt = self.entry['Prompt']
        result = False
        for i in range(3):
            inputs = self.call_chatgpt_fuzzing_tester(prompt)
            try:
                json_inputs = json.loads(inputs)
                result = True
                break
            except:
                continue
        if result:
            return json_inputs
        else:
            return {}


class InputMutatorAgent:
    def __init__(self, inputs, funname, code):
        self.inputs = inputs
        self.code = code
        self.funname = funname

    def mutate_value(self, value):
        """Mutates a single value based on its type."""
        if isinstance(value, bool):
            # Randomly flip the boolean value with a 50% chance
            return value if random.random() > 0.5 else not value
        if isinstance(value, int):
            # Mutate integers by adding or subtracting a random number
            return value + random.randint(-1000, 1000)
        elif isinstance(value, float):
            # Mutate floats by adding or subtracting a random float
            return value + random.uniform(-1000.0, 1000.0)
        elif isinstance(value, str):
            # Mutate strings by shuffling, adding random characters, or removing characters
            if len(value) == 0:
                return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 20)))
            mutation_type = random.choice(['shuffle', 'add', 'remove'])
            if mutation_type == 'shuffle':
                return ''.join(random.sample(value, len(value)))
            elif mutation_type == 'add':
                position = random.randint(0, len(value))
                return value[:position] + random.choice(string.ascii_letters + string.digits) + value[position:]
            elif mutation_type == 'remove' and len(value) > 1:
                position = random.randint(0, len(value)-1)
                return value[:position] + value[position+1:]
            else:
                return value
        elif isinstance(value, list):
            # Mutate all elements in the list
            return [self.mutate_value(element) for element in value]
        elif isinstance(value, dict):
            # Mutate dictionaries by mutating keys or values, adding or removing key-value pairs
            if len(value) == 0:
                # Add a new random key-value pair if dict is empty
                return {self.mutate_value(''): self.mutate_value('')}
            mutation_type = random.choice(
                ['mutate_key', 'mutate_value', 'add', 'remove'])
            if mutation_type == 'mutate_key':
                old_key = random.choice(list(value.keys()))
                new_key = self.mutate_value(old_key)
                value[new_key] = value.pop(old_key)
            elif mutation_type == 'mutate_value':
                key = random.choice(list(value.keys()))
                value[key] = self.mutate_value(value[key])
            elif mutation_type == 'add':
                value[self.mutate_value('')] = self.mutate_value('')
            elif mutation_type == 'remove' and len(value) > 1:
                key = random.choice(list(value.keys()))
                del value[key]
            return value
        else:
            return value  # For unsupported types, return the value as is

    def mutate(self, inputs):
        """Mutates the contents of the dynamic `inputs` object."""
        mutated_inputs = {}
        try:

            for key, value in inputs.items():
                mutated_inputs[key] = self.mutate_value(deepcopy(value))

        except AttributeError as e:
            print(f"Error: {e}. The `inputs` object is not a dictionary.")
            if isinstance(inputs, list):
                inputs = {i: item for i, item in enumerate(inputs)}
                for key, value in inputs.items():
                    mutated_inputs[key] = self.mutate_value(deepcopy(value))

        return mutated_inputs

    def fuzz_function(self, inputs, code, funname, num_tests=1):
        """Generates fuzzed inputs and runs the function with them."""
        # Extract and mutate the inputs
        return self.mutate(inputs)

    def mutate_inputs(self):

        mutated_inputs = self.fuzz_function(
            self.inputs, self.code, self.funname)
        return mutated_inputs
