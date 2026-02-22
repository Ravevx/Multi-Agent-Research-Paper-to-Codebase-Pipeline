import ast
import re
import json
from pathlib import Path
from crewai import Crew, Process

from crew_agents import (
    make_analyst_agent,
    make_architect_agent,
    make_coder_agent,
    make_reviewer_agent
)
from crew_tasks import (
    make_analysis_task,
    make_architecture_task,
    make_code_task,
    make_review_task
)
from project_planner import ProjectPlan
from config import PROJECT_OUTPUT_DIR
from paper_tools import PaperProcessor
from agent_logger import log_agent_output
from rag_store import build_rag, retrieve


# ─────────────────────────────────────────────
# Paper loading
# ─────────────────────────────────────────────

def load_paper_chunks(paper_path: str) -> list:
    processor = PaperProcessor()
    raw_pages = processor.load_paper(paper_path)
    chunks = [p.strip() for p in raw_pages if isinstance(p, str) and p.strip()]
    all_text = "\n\n".join(chunks)
    final_chunks = [all_text[i:i+800] for i in range(0, len(all_text), 700)]
    final_chunks = [c.strip() for c in final_chunks if len(c.strip()) > 20]
    print(f"Loaded {len(final_chunks)} chunks → RAG index")
    return final_chunks


def load_paper_rag(chunks: list):
    """Build RAG store once — reuse for all agent calls."""
    print("Building RAG index...")
    store = build_rag(chunks)
    print("RAG index ready")
    return store


# ─────────────────────────────────────────────
# JSON utilities
# ─────────────────────────────────────────────

def fix_json_strings(text: str) -> str:
    """Replace literal newlines/tabs inside JSON strings with spaces."""
    result = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            in_string = not in_string
            result.append(char)
        elif in_string and char in ('\n', '\t', '\r'):
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)


def safe_parse_json(raw: str) -> dict:
    """Multi-strategy JSON parser that handles common LLM JSON mistakes."""
    # Strategy 1: Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: json-repair library
    try:
        from json_repair import repair_json
        repaired = repair_json(raw)
        return json.loads(repaired)
    except Exception:
        pass

    # Strategy 3: Extract outermost {...} block
    try:
        start = raw.index('{')
        end = raw.rindex('}') + 1
        candidate = raw[start:end]
        return json.loads(candidate)
    except Exception:
        pass

    # Strategy 4: Fix unescaped newlines inside strings then retry
    try:
        fixed = re.sub(
            r'("(?:[^"\\]|\\.)*")',
            lambda m: m.group(0).replace('\n', ' ').replace('\t', ' '),
            raw
        )
        return json.loads(fixed)
    except Exception:
        pass

    return {}


# ─────────────────────────────────────────────
# Code cleaning
# ─────────────────────────────────────────────

def clean_code_output(raw_code: str) -> str:
    code = raw_code.strip()
    # Strip markdown fences
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```(?:\w+)?$", "", code).strip()

    # Strip leading explanation before first Python line
    match = re.search(r'^("""|\'\'\"|import |from |class |def |#)', code, re.MULTILINE)
    if match and match.start() > 0:
        print(f"  Stripped {match.start()} chars of leading explanation")
        code = code[match.start():]

    # Convert trailing prose to comments
    note_patterns = [
        r'\n\*\*Note[:\*]',
        r'\nNote:',
        r'\nThis implementation',
        r'\nThe code ',
        r'\nThe above ',
        r'\nFor a complete',
        r'\nIn a real',
        r'\nThis code ',
        r'\nThe implementation',
    ]
    for pattern in note_patterns:
        m = re.search(pattern, code, re.IGNORECASE)
        if m:
            prose = code[m.start():]
            commented = "\n" + "\n".join(
                f"# {line}" if line.strip() and not line.strip().startswith("#") else line
                for line in prose.strip().splitlines()
            )
            code = code[:m.start()] + commented
            print(f"  Converted trailing note to comments")
            break

    return code.strip()


# ─────────────────────────────────────────────
# Fallbacks
# ─────────────────────────────────────────────

def _fallback_plan(description: str) -> dict:
    return {
        "project_name": "research-implementation",
        "description": description.replace("\n", " ")[:100],
        "folders": ["src"],
        "files": [
            {
                "filename": "src/model.py",
                "purpose": "Core model from paper",
                "dependencies": [],
                "classes": ["Model"],
                "functions": [],
                "logic_summary": "Core implementation from paper.",
                "class_details": {"Model": "Main model class from paper"},
                "function_details": {}
            },
            {
                "filename": "src/main.py",
                "purpose": "Entry point — imports Model and runs end-to-end demo",
                "dependencies": [],
                "classes": [],
                "functions": ["main"],
                "logic_summary": "Instantiates Model and runs a forward pass with sample data.",
                "class_details": {},
                "function_details": {
                    "main": "Imports Model from src.model; defines config; instantiates Model; runs forward pass; prints output"
                }
            }
        ],
        "dependencies": [],
        "main_file": "src/main.py",
        "readme_content": "Research paper Python implementation."
    }


def _ask_llm_simple_plan(analysis: str) -> dict:
    from llm import getllm
    print("Calling LLM with simplified schema...")
    llm = getllm(temperature=0.0)
    prompt = f"""You are a software architect. Based on the paper analysis below, return a JSON project plan.

PAPER ANALYSIS:
{analysis}

RULES (ALL mandatory):
- Return ONLY valid JSON — nothing else, no markdown, no explanation, no backticks
- Every string value must be ONE line — no newlines inside any string
- dependencies: pip package names only — no version numbers (write "torch" not "torch==1.7")
- readme_content: one short sentence — no commas at all
- Derive ALL filenames, class names, function names from the paper analysis
- ALWAYS include "src/main.py" as the last file
- main_file must be exactly "src/main.py"
- root dependencies must include ALL pip packages from all files — never empty
- Do NOT include algorithm_steps in any file

JSON SCHEMA:
{{
  "project_name": "",
  "description": "",
  "folders": ["src"],
  "files": [
    {{
      "filename": "src/component.py",
      "purpose": "",
      "dependencies": ["torch"],
      "classes": ["ClassName"],
      "functions": [],
      "logic_summary": "",
      "class_details": {{"ClassName": "description"}},
      "function_details": {{}}
    }},
    {{
      "filename": "src/main.py",
      "purpose": "Entry point that imports all components and runs end-to-end demo",
      "dependencies": [],
      "classes": [],
      "functions": ["main"],
      "logic_summary": "Wires all components together and runs a full forward pass",
      "class_details": {{}},
      "function_details": {{"main": "Imports every class from every other file; instantiates each; runs full pipeline"}}
    }}
  ],
  "dependencies": ["torch", "numpy"],
  "main_file": "src/main.py",
  "readme_content": "Short description of the project and how to run it"
}}

Return ONLY the JSON. No explanation before or after."""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
        text = fix_json_strings(text)
        start = text.index('{')
        end = text.rindex('}') + 1
        candidate = text[start:end]
        parsed = safe_parse_json(candidate)
        if parsed and parsed.get("files"):
            print(f"  LLM re-plan succeeded: {len(parsed['files'])} files")
            return parsed
        print("  LLM re-plan returned empty or invalid JSON")
        return {}
    except Exception as e:
        print(f"  LLM re-plan failed: {e}")
        return {}


# ─────────────────────────────────────────────
# Step 1: Analyse paper
# ─────────────────────────────────────────────

def analyse_paper(rag_store) -> str:
    print("\nStep 1: Analyst reading paper...")
    context = retrieve(
        rag_store,
        "What is this paper about? Key components, algorithms, equations, architecture, data flow, inputs, outputs.",
        k=10
    )
    print(f"  RAG context for analyst: {len(context)} chars")
    analyst = make_analyst_agent()
    task = make_analysis_task(context, analyst)
    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)
    result = str(crew.kickoff()).strip()
    print(f"\nANALYSIS OUTPUT (first 500):\n{result[:500]}\n")
    log_agent_output("_latest_run", "analyst", result)
    return result


# ─────────────────────────────────────────────
# Step 2: Generate project plan
# ─────────────────────────────────────────────

def _enforce_plan_invariants(plan_dict: dict) -> dict:
    """
    Post-parse corrections that must always hold regardless of what the LLM produced.
    Call this after ANY parse path (success, re-plan, or fallback).
    """
    # 1. main_file is always src/main.py
    plan_dict["main_file"] = "src/main.py"

    # 2. Root dependencies must include ALL per-file deps — never empty
    all_deps: list[str] = list(plan_dict.get("dependencies", []))
    for f in plan_dict.get("files", []):
        for dep in f.get("dependencies", []):
            if dep and dep not in all_deps:
                all_deps.append(dep)
    plan_dict["dependencies"] = all_deps

    # 3. Set defaults for all file fields
    for f in plan_dict.get("files", []):
        f.setdefault("classes", [])
        f.setdefault("functions", [])
        f.setdefault("logic_summary", f.get("purpose", ""))
        f.setdefault("class_details", {})
        f.setdefault("function_details", {})
        # Remove algorithm_steps if present (we no longer use them)
        f.pop("algorithm_steps", None)

    # 4. Ensure there is always an entry point
    filenames = [f["filename"] for f in plan_dict.get("files", [])]
    has_entry = any("main" in fn or "train" in fn for fn in filenames)
    if not has_entry:
        print("  No entry point found — auto-adding src/main.py")
        # Collect all class names defined in other files
        all_classes = []
        for f in plan_dict.get("files", []):
            all_classes.extend(f.get("classes", []))
        classes_desc = ", ".join(all_classes) if all_classes else "all components"
        plan_dict["files"].append({
            "filename": "src/main.py",
            "purpose": "Entry point — imports all components and runs end-to-end demo with sample data",
            "dependencies": [],
            "classes": [],
            "functions": ["main"],
            "logic_summary": "Imports all modules, instantiates every component, runs a full forward pass, prints output shapes",
            "class_details": {},
            "function_details": {
                "main": (
                    f"Imports {classes_desc} from their respective modules; "
                    "defines config dict with all paper hyperparameters; "
                    "instantiates each component in dependency order; "
                    "generates sample input tensors with correct shapes; "
                    "runs full pipeline end-to-end; prints output shapes"
                )
            }
        })

    # 5. Ensure src/main.py is always the LAST file
    files = plan_dict.get("files", [])
    main_files = [f for f in files if f["filename"] == "src/main.py"]
    other_files = [f for f in files if f["filename"] != "src/main.py"]
    if main_files:
        plan_dict["files"] = other_files + main_files
    else:
        # Build one from scratch
        all_classes = []
        for f in other_files:
            all_classes.extend(f.get("classes", []))
        classes_desc = ", ".join(all_classes) if all_classes else "all components"
        plan_dict["files"] = other_files + [{
            "filename": "src/main.py",
            "purpose": "Entry point — imports all components and runs end-to-end demo with sample data",
            "dependencies": [],
            "classes": [],
            "functions": ["main"],
            "logic_summary": "Imports all modules, instantiates every component, runs a full forward pass, prints output shapes",
            "class_details": {},
            "function_details": {
                "main": (
                    f"Imports {classes_desc} from their respective modules; "
                    "defines config dict with all paper hyperparameters; "
                    "instantiates each component in dependency order; "
                    "generates sample input tensors with correct shapes; "
                    "runs full pipeline end-to-end; prints output shapes"
                )
            }
        }]

    # 6. Fix main.py: classes[] must be [] — it only imports, not redefines
    for f in plan_dict["files"]:
        if f["filename"] == "src/main.py":
            f["classes"] = []
            f["class_details"] = {}
            if "main" not in f.get("functions", []):
                f["functions"] = ["main"]

    # 7. Other defaults
    plan_dict.setdefault("readme_content", "Research paper implementation.")
    plan_dict.setdefault("folders", ["src"])

    return plan_dict


def generate_plan(analysis: str) -> ProjectPlan:
    print("\nStep 2: Architect designing project structure...")
    architect = make_architect_agent()
    task = make_architecture_task(analysis, architect)
    crew = Crew(agents=[architect], tasks=[task], process=Process.sequential, verbose=False)
    result = str(crew.kickoff()).strip()
    print(f"\nARCHITECT FULL OUTPUT:\n{result}\n")
    log_agent_output("_latest_run", "architect", result)

    # Clean raw output
    raw = re.sub(r"^```(?:json)?\n?", "", result)
    raw = re.sub(r"\n?```(?:json)?$", "", raw).strip()
    raw = fix_json_strings(raw)

    # Fix unquoted readme_content values (rare LLM mistake)
    raw = re.sub(
        r'"readme_content":\s*(?!")([^,}\]]+)',
        lambda m: f'"readme_content": "{m.group(1).strip()}"',
        raw
    )

    # Extract outermost {...} block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
        print(f"  Extracted JSON block ({len(raw)} chars)")

    # Multi-strategy parse
    plan_dict = safe_parse_json(raw)

    if not plan_dict or not plan_dict.get("files"):
        print("  JSON parse failed or no files — asking LLM for re-plan...")
        plan_dict = _ask_llm_simple_plan(analysis)

    if not plan_dict or not plan_dict.get("files"):
        print("  LLM re-plan also failed — using minimal fallback...")
        plan_dict = _fallback_plan(analysis[:150])

    print(f"  Raw plan: {len(plan_dict.get('files', []))} files")

    # Always enforce invariants regardless of parse path
    plan_dict = _enforce_plan_invariants(plan_dict)

    print(f"  Final plan: {len(plan_dict.get('files', []))} files, {len(plan_dict.get('dependencies', []))} root deps")

    try:
        log_agent_output("_latest_run", "plan_json", json.dumps(plan_dict, indent=2))
    except Exception:
        pass

    return ProjectPlan(**plan_dict)


# ─────────────────────────────────────────────
# Step 3: Generate code per file
# ─────────────────────────────────────────────

def generate_code(plan: ProjectPlan, analysis: str, rag_store) -> dict:
    print("\nStep 3: Coder writing files...")
    coder = make_coder_agent()
    reviewer = make_reviewer_agent()
    files_written = {}

    for file_spec in plan.files:
        print(f"\n{'='*60}")
        print(f"  Writing: {file_spec.filename}")

        # RAG: retrieve context relevant to this specific file
        query = (
            f"Technical details to implement: "
            f"{', '.join(file_spec.classes + file_spec.functions)}. "
            f"Include equations, tensor dimensions, hyperparameters."
        )
        relevant_context = retrieve(rag_store, query, k=5)
        print(f"  RAG context: {len(relevant_context)} chars")

        # ── Code generation ──
        code_task = make_code_task(file_spec.model_dump(), relevant_context, coder)
        code_crew = Crew(agents=[coder], tasks=[code_task], process=Process.sequential, verbose=False)
        raw_code = str(code_crew.kickoff()).strip()
        print(f"  Raw chars: {len(raw_code)}")

        code = clean_code_output(raw_code)
        print(f"  Cleaned: {len(code.splitlines())} lines")

        # ── Syntax check ──
        try:
            ast.parse(code)
            print(f"  ✓ Syntax OK")
        except SyntaxError as e:
            print(f"  ✗ Syntax error line {e.lineno}: {e.msg}")
            code = f"# Syntax error detected line {e.lineno}: {e.msg}\n" + code

        # ── Reviewer pass ──
        try:
            review_task = make_review_task(file_spec.filename, code, reviewer)
            review_crew = Crew(
                agents=[reviewer], tasks=[review_task],
                process=Process.sequential, verbose=False
            )
            review_raw = str(review_crew.kickoff()).strip()
            review_raw = re.sub(r"^```(?:json)?\n?", "", review_raw)
            review_raw = re.sub(r"\n?```$", "", review_raw).strip()
            review = safe_parse_json(review_raw)
            if review:
                status = review.get("status", "unknown")
                issues = review.get("issues", [])
                suggestion = review.get("suggestion", "")
                print(f"  Review: {status}")
                if issues:
                    print(f"  Issues: {issues}")
                if suggestion:
                    print(f"  Suggestion: {suggestion}")
                log_agent_output(
                    plan.project_name,
                    f"review_{file_spec.filename.replace('/', '_')}",
                    json.dumps(review, indent=2)
                )
            else:
                print(f"  Review parse failed — skipping")
        except Exception as e:
            print(f"  Reviewer error (non-fatal): {e}")

        log_agent_output(
            plan.project_name,
            f"coder_{file_spec.filename.replace('/', '_')}",
            code,
            extra=f"Raw chars: {len(raw_code)} | Cleaned lines: {len(code.splitlines())}"
        )

        files_written[file_spec.filename] = code

    return files_written


# ─────────────────────────────────────────────
# Step 4: Save project to disk
# ─────────────────────────────────────────────

_saved_projects: set = set()


def save_project(plan: ProjectPlan, files_written: dict) -> str:
    project_dir = Path(PROJECT_OUTPUT_DIR) / plan.project_name
    already_saved = str(project_dir) in _saved_projects

    def log(msg):
        if not already_saved:
            print(msg)

    log(f"\nSaving to: {project_dir}")

    for filename, code in files_written.items():
        filepath = project_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(code, encoding="utf-8")
        log(f"  {filepath} ({len(code.splitlines())} lines)")

    # Write requirements.txt
    (project_dir / "requirements.txt").write_text(
        "\n".join(plan.dependencies), encoding="utf-8"
    )

    # Write README.md
    (project_dir / "README.md").write_text(
        plan.readme_content, encoding="utf-8"
    )

    log(f"Done: {project_dir}")
    _saved_projects.add(str(project_dir))
    return str(project_dir)
