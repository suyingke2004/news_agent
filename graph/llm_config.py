import os

from langchain_openai import ChatOpenAI

MODEL_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-4o",
        "base_url": None,
    },
    "deepseek": {
        "name": "DeepSeek",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "zhipu": {
        "name": "Zhipu AI",
        "default_model": "glm-4",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    },
    "ali": {
        "name": "Alibaba Cloud",
        "default_model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "moonshot": {
        "name": "Moonshot AI",
        "default_model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
    },
}


def get_llm(
    provider: str = "deepseek",
    model_name: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the specified provider.

    All providers use the OpenAI-compatible API format via base_url.
    """
    provider_info = MODEL_PROVIDERS.get(provider)
    if not provider_info:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"Missing {provider.upper()}_API_KEY in environment")

    model = model_name or provider_info["default_model"]

    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
    }

    if provider_info["base_url"]:
        kwargs["base_url"] = provider_info["base_url"]

    return ChatOpenAI(**kwargs)
