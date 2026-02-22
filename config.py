import os
from dotenv import load_dotenv
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
LM_MODEL = os.getenv("LM_MODEL", "ministral-3-3b-instruct-2512")

# Aliases for compatibility
LMSTUDIOURL = LMSTUDIO_URL
LMMODEL = LM_MODEL


MAX_FILESIZE = 8000
MAX_FILES = 2
TEMP_DIR = "temp"
PAPER_OUTPUT_DIR = "output/papers"
PROJECT_OUTPUT_DIR = "output/projects"

# Create all folders
for d in [TEMP_DIR, PAPER_OUTPUT_DIR, PROJECT_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Config loaded: LMSTUDIO_URL={LMSTUDIO_URL}")
print(f"Output dirs ready: {PAPER_OUTPUT_DIR}, {PROJECT_OUTPUT_DIR}")
