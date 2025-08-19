from llms import LLMInterface


class LLMDecisionAgent():
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    # test_choices = ['unit_test', 'static_analysis', 'fuzzing']

    def decide(self, prompt, code, choice) -> str:
        prompt = f'''**Requirement**:
{prompt}
**Code**:
{code}
**Instruction**:
You now have the following test tools to choose from: {str(choice)}
If you think the code needs to perform relevant tests, please output the name of the test tool, otherwise output "none".
If a test is required, only output the name of the highest priority test tool. Only One Test Tool.
*Note: If you think the code is already perfect, you can output "none" directly. Else, you can output the name of the test tool you think is most suitable.
Your choice:
''' 
        for i in range(3):
            try:
                result = self.llm.generate(prompt)
                if result.startswith('unit_test'):
                    return 'unit_test'
                elif result.startswith:
                    return 'static_analysis'
                elif result.startswith('fuzzing'):
                    return 'fuz zing'
                else:
                    return 'none'
            except Exception as e:
                print(f"Error generating response: {e}")
        return 'none'

    