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
        prompt_path = "./prompts_fewshot/coder_agent_prompt.txt"
        with open(prompt_path, "r") as f:
            self.construct_few_shot_prompt = f.read()

    def call_chatgpt_programmer(self, prompt):
        text = 'Please only respond with the code in Python as the output. Code should in triple backticks format(i.e., in ```python ```)\n' + prompt
        try:
            completion = self.llm.generate(text)
            completion = completion.strip()
            completion = preprocess_string(completion, "python")

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def call_chatgpt_programmer_feedback_static(self, completion, entry, cwe_code, issue_text):

        lg = "python"
        if issue_text:
            gpt_prompt = (
                "Please modify the code to fix the following security vulnerability.\n" +
                "**Note: No need to generate test cases or test code(e.g., assert )\n" +
                f"\nHere is the code that has been detected to have a security vulnerability:\n```{lg}\n" +
                completion +
                f"\n```\nThe CWE code and details of the vulnerability detected are the following:\n {cwe_code} - {issue_text}" +
                f"\n```\nPlease modify the code so that it does not have any security vulnerability. The re-completion code should in triple backticks format(i.e., in ```{lg} ```)."
            )
        else:
            gpt_prompt = (
                "Please modify the code to fix the following security vulnerability.\n" +
                "**Note: No need to generate test cases or test code(e.g., assert )\n" +
                f"\nHere is the code that has been detected to have a security vulnerability:\n```{lg}\n" +
                completion +
                f"\n```\nThe CWE code and details of the vulnerability detected are the following:\n {cwe_code}" +
                f"\n```\nPlease modify the code so that it does not have any security vulnerability but keeps main functionality. The re-completion code should in triple backticks format(i.e., in ```{lg} ```)."
            )
        try:
            completion = self.llm.generate(
                "You are a code developer assistant." + gpt_prompt)
            completion = completion.strip()
            completion = preprocess_string(completion, "python")

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def call_chatgpt_programmer_feedback_fuzzing(self, completion, entry, fuzzing_feedback):

        output_string = fuzzing_feedback
        lg = "python"
        gpt_prompt = (
            "Please re-completion the code to fix the error message.\n" +
            "**Note: No need to generate test cases or test code(e.g., assert )\n" +
            f"\nHere is the previous version:\n```{lg}\n" +
            completion +
            f"\n```\nWhen calling the function with the following inputs, it raises errors. The inputs and errors are the following:\n" +
            output_string +
            f"\n```\nPlease fix the bugs and return the code. The re-completion code should in triple backticks format(i.e., in ```{lg} ```)."
        )
        try:
            completion = self.llm.generate(
                "You are a code developer assistant."+gpt_prompt)
            completion = completion.strip()
            completion = preprocess_string(completion, "python")

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def call_chatgpt_programmer_feedback_func(self, completion, inputs):
        lg = "python"
        gpt_prompt = (
            "Please re-completion the code to fix the error message.\n" +
            "**Note: No need to generate test cases or test code(e.g., assert )\n" +
            f"\nHere is the previous version:\n```{lg}\n" +
            completion +
            f"\n```\nWhen calling the function with the following inputs, it raises errors. The inputs and errors are the following:\n" +
            inputs +
            f"\n```\nPlease fix the bugs and only return the fixed code. The re-completion code should in triple backticks format(i.e., in ```{lg} ```)."
        )
        try:
            completion = self.llm.generate(
                "You are a code developer assistant. "+gpt_prompt)
            completion = completion.strip()
            completion = preprocess_string(completion, "python")

        except Exception as e:
            print(e)
            completion = ""

        return completion

    def write_code(self,):
        prompt = self.entry['Prompt']
        code = self.call_chatgpt_programmer(prompt)
        return code

    def write_code_feedback_static(self, completion, cwe_code, issue_text):
        code = self.call_chatgpt_programmer_feedback_static(
            completion, self.entry, cwe_code, issue_text)
        return code

    def write_code_feedback_fuzz(self, completion, fuzzing_feedback):
        code = self.call_chatgpt_programmer_feedback_fuzzing(
            completion, self.entry, fuzzing_feedback)
        return code

    def write_code_feedback_func(self, completion, inputs):
        code = self.call_chatgpt_programmer_feedback_func(
            completion, inputs)
        return code
