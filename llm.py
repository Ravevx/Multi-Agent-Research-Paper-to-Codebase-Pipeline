# llm.py
from langchain_openai import ChatOpenAI
from config import LMSTUDIOURL, LMMODEL

def getllm(temperature: float = 0.1):
    return ChatOpenAI(
        base_url=LMSTUDIOURL,
        api_key="lm-studio",
        model=LMMODEL,
        temperature=temperature,
        max_tokens=8192,
        timeout=300,
    )
