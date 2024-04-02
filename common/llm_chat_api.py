# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: llm_chat_api.py
# @time: 2023/7/22 15:12
import json
import requests
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from config.config_parser import (EMBEDDING_API,
                                  ZHIPU_API_KEY,
                                  CHAT_COMPLETION_API,
                                  SYSTEM_ROLE,
                                  OPENAI_KEY,
                                  OPENAI_EMBEDDING_API,
                                  ZHIPU_EMBEDDING_API,
                                  OPENAI_CHAT_COMPLETION_API)
from utils.logger import logger
from config.tools import tools

def get_text_embedding(req_text, model_name):
    if model_name == 'embedding-2':
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=ZHIPU_API_KEY) 
        response = client.embeddings.create(
            model=model_name, #填写需要调用的模型名称
            input=req_text
        )
        return json.loads(response.model_dump_json())["data"][0]["embedding"]
    else:
        headers = {'Content-Type': 'application/json'}
        embedding_api = EMBEDDING_API
        if "gpt" in model_name:
            headers["Authorization"] = f"Bearer {OPENAI_KEY}"
            model_name = "text-embedding-ada-002"
            embedding_api = OPENAI_EMBEDDING_API
        payload = json.dumps({"model": model_name, "input": req_text})
        new_req = requests.request("POST", embedding_api, headers=headers, data=payload)
        return new_req.json()['data'][0]['embedding']


def chat_completion(message, model_name):
    if model_name =="glm-4":
        payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_ROLE
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0.8,
        "max_tokens": 300
        }
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=ZHIPU_API_KEY) # 填写您自己的APIKey
        response = client.chat.completions.create(
            **payload,
            tools=tools
        )
        logger.info(f"model_name: {model_name}, token use: {response.usage.total_tokens}")
        return response.choices[0].message.content
    else:
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_ROLE
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            "temperature": 0,
            "max_tokens": 300
        })
        chat_completion_api = CHAT_COMPLETION_API
        headers = {'Content-Type': 'application/json'}
        if "gpt" in model_name:
            headers["Authorization"] = f"Bearer {OPENAI_KEY}"
            chat_completion_api = OPENAI_CHAT_COMPLETION_API
        response = requests.request("POST", chat_completion_api, headers=headers, data=payload)
        logger.info(f"model_name: {model_name}, response: {response.text}")
        return response.json()['choices'][0]['message']['content']

