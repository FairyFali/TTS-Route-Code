from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif ('gpt' in model_name.lower() or 'deepseek' in model_name.lower() or 'doubao' in model_name.lower()): # any version of GPTChat like "gpt-4-1106-preview"
            print('### get model from gpt api.')
            model = cls.registry.get('GPTChat', model_name)
        elif 'PRM' in model_name:
            model = cls.registry.get('PRM', model_name)
        elif model_name.lower().startswith('openrouter:') or '/' in model_name:
            # OpenRouter model ids use the "provider/model" form, e.g.
            # "qwen/qwen-2.5-7b-instruct". An explicit "openrouter:" prefix is
            # also accepted and stripped before being sent to the API.
            print('### get model from openrouter.')
            real_name = model_name[len('openrouter:'):] if model_name.lower().startswith('openrouter:') else model_name
            model = cls.registry.get('openrouter', real_name)
        else:
            print('### get model from ollama.')
            model = cls.registry.get('ollama', model_name)

        return model
