import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional
from dotenv import load_dotenv
import async_timeout
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.price import cost_count
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry


# OpenRouter exposes an OpenAI-compatible endpoint.
OPENROUTER_URL = "https://openrouter.ai/api/v1"

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Optional attribution headers (safe to leave as-is or blank).
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/metauto-ai/GPTSwarm",
    "X-Title": "GPTSwarm",
}


def openrouter_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return ''

    api_kwargs: Dict[str, Any]
    api_kwargs = dict(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY,
                      default_headers=OPENROUTER_HEADERS)
    client = OpenAI(**api_kwargs)

    formated_messages = [asdict(message) for message in messages]
    response = client.chat.completions.create(model=model,
    messages=formated_messages,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    n=num_comps)

    if num_comps == 1:
        cost_count(response, model)
        return response.choices[0].message.content

    cost_count(response, model)

    return [choice.message.content for choice in response.choices]


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def openrouter_achat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=True,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return ''

    api_kwargs: Dict[str, Any]
    api_kwargs = dict(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY,
                      default_headers=OPENROUTER_HEADERS)
    aclient = AsyncOpenAI(**api_kwargs)

    formated_messages = [asdict(message) for message in messages]
    try:
        async with async_timeout.timeout(1000):
            response = await aclient.chat.completions.create(model=model,
            messages=formated_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps)
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("OpenRouter Timeout")
    if num_comps == 1:
        price, prompt_len, completion_len = cost_count(response, model)
        return response.choices[0].message.content, prompt_len, completion_len, price

    # n>1: surface usage totals too (prefill shared, decode summed across comps)
    price, prompt_len, completion_len = cost_count(response, model)
    return ([choice.message.content for choice in response.choices],
            prompt_len, completion_len, price)


@LLMRegistry.register('openrouter')
class openrouter(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        print('### Initialize an OpenRouter LLM', model_name)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await openrouter_achat(self.model_name,
                                      messages,
                                      max_tokens,
                                      temperature,
                                      num_comps)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return openrouter_chat(self.model_name,
                               messages,
                               max_tokens,
                               temperature,
                               num_comps)
