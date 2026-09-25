#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.utils.log import swarmlog
from swarm.utils.globals import Cost, PromptTokens, CompletionTokens

# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing

def cost_count(response, model_name):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    if "gpt-4" in model_name:
        try:
            branch = "gpt-4"
            prompt_len = response.usage.prompt_tokens
            completion_len = response.usage.completion_tokens
            price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
                completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
        except:
            branch = "gpt-4"
            prompt_len = response["usage"]["prompt_tokens"]
            completion_len = response["usage"]["completion_tokens"]
            price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
                completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
    elif "gpt-3.5" in model_name:
        branch = "gpt-3.5"
        prompt_len = response.usage.prompt_tokens
        completion_len = response.usage.completion_tokens
        price = prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] /1000 + \
            completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] /1000
    elif "dall-e" in model_name:
        branch = "dall-e"
        price = 0.0
        prompt_len = 0
        completion_len = 0
    elif "deepseek" in model_name or "doubao" in model_name:
        branch = "deepseek"
        try:
            prompt_len = response.usage.prompt_tokens
            completion_len = response.usage.completion_tokens
        except:
            prompt_len = response["usage"]["prompt_tokens"]
            completion_len = response["usage"]["completion_tokens"]

        # deepseek 价格：单位 元 / 千 tokens
        if model_name == "deepseek-v3":
            input_price = 0.0020   # 输入价格
            output_price = 0.0080  # 输出价格
        elif model_name == "deepseek-r1":
            input_price = 0.0040   # 输入价格
            output_price = 0.0160  # 输出价格
        elif model_name == "doubao-1.5-pro":
            input_price = 0.0008   # 输入价格
            output_price = 0.002  # 输出价格
        else:
            # 未知 deepseek 型号，费用记 0
            input_price = 0.0
            output_price = 0.0

        price = prompt_len * input_price / 1000 + completion_len * output_price / 1000
        price = price * 0.1406
    elif "llama" in model_name or "gemma" in model_name or "qwen" in model_name:
        try:
            prompt_len = response.usage.prompt_tokens
            completion_len = response.usage.completion_tokens
        except:
            prompt_len = response["usage"]["prompt_tokens"]
            completion_len = response["usage"]["completion_tokens"]

        if model_name in ("llama3.2-1b-longcontext:latest","gemma3-1b-longcontext:latest","qwen2.5-1.5b-longcontext:latest") :
            input_price = 0.02  # 输入价格
            output_price = 0.02 # 输出价格
        elif model_name in ("llama3.2-3b-longcontext:latest","gemma1-2b-longcontext:latest","qwen2.5-3b-longcontext:latest") :
            input_price = 0.06   # 输入价格
            output_price = 0.06  # 输出价格
        elif model_name in ("llama3.1-8b-longcontext:latest","qwen2.5-7b-longcontext:latest") :
            input_price = 0.18   # 输入价格
            output_price = 0.18  # 输出价格
        elif model_name == "gemma1-7b-longcontext:latest":
            input_price = 0.27   # 输入价格
            output_price = 0.27
        elif model_name == "llama3.1-70b-longcontext:latest":
            input_price = 0.88   # 输入价格
            output_price = 0.88
        else:
            input_price = 0  # 输入价格
            output_price = 0
        
        price = prompt_len * input_price / 1000000 + completion_len * output_price / 1000000

    else:
        branch = "other"
        price = 0.0
        prompt_len = response.usage.prompt_tokens
        completion_len = response.usage.completion_tokens

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

OPENAI_MODEL_INFO ={
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {
            "context window": 128000, 
            "training": "Jan 2024", 
            "input": 0.01, 
            "output": 0.03
        },      
        "gpt-4-1106-preview": {
            "context window": 128000, 
            "training": "Apr 2023", 
            "input": 0.01, 
            "output": 0.03
        },
        "gpt-4-vision-preview": {
            "context window": 128000, 
            "training": "Apr 2023", 
            "input": 0.01, 
            "output": 0.03
        },
        "gpt-4": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.03, 
            "output": 0.06
        },
        "gpt-4-0314": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.03, 
            "output": 0.06
        },
        "gpt-4-32k": {
            "context window": 32768, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4-32k-0314": {
            "context window": 32768, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        },
        "gpt-4-0613": {
            "context window": 8192, 
            "training": "Sep 2021", 
            "input": 0.06, 
            "output": 0.12
        }
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125": {
            "context window": 16385, 
            "training": "Jan 2024", 
            "input": 0.0010, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-1106": {
            "context window": 16385, 
            "training": "Sep 2021", 
            "input": 0.0010, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-instruct": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-0301": {
            "context window": 4096, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-0613": {
            "context window": 16384, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        },
        "gpt-3.5-turbo-16k-0613": {
            "context window": 16384, 
            "training": "Sep 2021", 
            "input": 0.0015, 
            "output": 0.0020
        }
    },
    "dall-e": {
        "current_recommended": "dall-e-3",
        "dall-e-3": {
            "release": "Nov 2023",
            "standard": {
                "1024×1024": 0.040,
                "1024×1792": 0.080,
                "1792×1024": 0.080
            },
            "hd": {
                "1024×1024": 0.080,
                "1024×1792": 0.120,
                "1792×1024": 0.120
            }
        },
        "dall-e-2": {
            "release": "Nov 2022",
            "1024×1024": 0.020,
            "512×512": 0.018,
            "256×256": 0.016
        }
    }
}



