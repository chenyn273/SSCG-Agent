from abc import ABC, abstractmethod
import json
import os
from openai import OpenAI
import dashscope
import requests


class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        根据用户需求生成代码
        :param prompt: 用户需求的文本描述
        :return: 生成的Python代码
        """
        pass


class OpenAI_LLM(LLMInterface):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://xiaoai.plus/v1",
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        times = 2
        for i in range(times):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return completion.choices[0].message.content
            except Exception as e:
                print('API ERROR')


class DeepSeek_LLM(LLMInterface):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        times = 2
        for i in range(times):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ]
                )
                return completion.choices[0].message.content
            except Exception as e:
                print('API ERROR')


class Qwen_LLM(LLMInterface):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        times = 2
        for i in range(times):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system',
                            'content': 'You are a professional Python developer.'},
                        {'role': 'user', 'content': prompt}],
                )
                return completion.choices[0].message.content
            except Exception as e:
                print('API ERROR')


class GuijiFlow(LLMInterface):

    def __init__(self, api_key: str, model: str):
        pass

    def generate(self, prompt: str) -> str:
        times = 2
        for i in range(times):
            try:
                url = "https://api.siliconflow.cn/v1/chat/completions"
                payload = {
                    "model": "deepseek-ai/DeepSeek-V2.5",
                    "stream": False,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "top_p": 0.7,
                    "top_k": 50,
                    "frequency_penalty": 0.5,
                    "n": 1,
                    "stop": [],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                headers = {
                    "Authorization": "Bearer sk-asztzxczyeuvusdqbmxsliwdorchtayfhxerwjoavstwzcem",
                    "Content-Type": "application/json"
                }
                response = requests.request(
                    "POST", url, json=payload, headers=headers)
                response = json.loads(response.text)
                # 提取 content 字段的内容
                content = response['choices'][0]['message']['content']
                return content
            except Exception as e:
                print('API ERROR')
