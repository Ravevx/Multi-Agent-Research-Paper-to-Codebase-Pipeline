import os
from datetime import datetime
from pathlib import Path
from config import PROJECT_OUTPUT_DIR


def get_log_path(project_name: str) -> Path:
    log_dir = Path(PROJECT_OUTPUT_DIR) / project_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def log_agent_output(project_name: str, agent: str, content: str, extra: str = ""):
    """
    Append one agent's output to logs/<project_name>/agent_outputs.txt
    agent: one of 'analyst', 'architect', 'coder_<filename>', 'plan_json'
    """
    log_path = get_log_path(project_name) / "agent_outputs.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 70

    entry = (
        f"\n{separator}\n"
        f"[{timestamp}] AGENT: {agent.upper()}\n"
        f"{separator}\n"
        f"{content.strip()}\n"
    )
    if extra:
        entry += f"\n--- EXTRA INFO ---\n{extra.strip()}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"Logged {agent} output → {log_path}")
