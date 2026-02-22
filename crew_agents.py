from crewai import Agent, LLM
from config import LMSTUDIOURL, LMMODEL
import os

os.environ["OPENAI_API_KEY"] = "lm-studio"


def _base_llm(temperature: float = 0.0, max_tokens: int = 8192) -> LLM:
    return LLM(
        model=f"lm_studio/{LMMODEL}",
        base_url=LMSTUDIOURL,
        api_key="lm-studio",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def make_analyst_agent() -> Agent:
    return Agent(
        role="Research Paper Analyst",
        goal="Extract every technical detail from a research paper and produce a complete, confident technical breakdown",
        backstory=(
            "You are an expert research scientist who reads academic papers across all domains of AI, ML, and "
            "data science. You always treat provided content as a complete paper and extract the exact algorithms, "
            "data flows, mathematical operations, and implementation details. You never hedge or say content is "
            "incomplete — you extract maximum technical value from whatever is given and state findings as facts."
        ),
        llm=_base_llm(temperature=0.0, max_tokens=8192),
        verbose=False
    )


def make_architect_agent() -> Agent:
    return Agent(
        role="Software Architect",
        goal="Design a clean modular Python project structure from a paper analysis and return ONLY valid compact JSON",
        backstory=(
            "You are a senior software architect who specialises in translating research paper analyses into clean, "
            "well-structured Python projects. You decide which files, classes, and functions are needed, what "
            "libraries to use, and how the components connect. You always return ONLY the JSON object — no "
            "explanation, no markdown, no preamble. Every string you write fits on a single line."
        ),
        # Architect needs more headroom because it produces a multi-file JSON plan.
        # If Mistral 3B context allows it, push this up; 12288 is a safe ceiling for most 3B deployments.
        llm=_base_llm(temperature=0.0, max_tokens=12288),
        verbose=False
    )


def make_coder_agent() -> Agent:
    return Agent(
        role="Research Implementation Engineer",
        goal="Write complete, production-quality Python code that exactly replicates a research paper component",
        backstory=(
            "You are an expert Python engineer who specialises in implementing research papers. Given a precise "
            "file specification and technical context, you write complete, working code with proper imports, type "
            "hints, and inline comments referencing paper equations. You never write placeholders, stubs, or `pass` "
            "statements. You output ONLY valid Python source code — nothing before the imports, nothing after the "
            "last line of code."
        ),
        llm=_base_llm(temperature=0.0, max_tokens=8192),
        verbose=False
    )


def make_reviewer_agent() -> Agent:
    return Agent(
        role="Code Quality Reviewer",
        goal="Verify generated code is complete, syntactically correct, and matches the specification; return a JSON report",
        backstory=(
            "You are a Python expert and ML researcher who reviews code implementations of research papers. "
            "You check that all required components are implemented correctly and that no stubs or placeholders "
            "remain. You always return ONLY the JSON review report — no explanation outside the JSON."
        ),
        llm=_base_llm(temperature=0.0, max_tokens=2048),
        verbose=False
    )
