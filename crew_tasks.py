from crewai import Task


def make_analysis_task(context: str, agent) -> Task:
    return Task(
        description=f"""Read this research paper content and produce a COMPLETE technical analysis.

INSTRUCTIONS — READ CAREFULLY:
- Start your response immediately with "## 1. WHAT IS THIS PAPER ABOUT?" — no preamble, no introduction
- Treat all provided content as a complete paper. Extract maximum technical detail from exactly what is given. State all findings as facts.
- State all findings directly. Example: "The paper proposes X" — not "this might be about X" or "assuming this is from..."
- Write each section ONCE. Do NOT add a Final Answer, summary, or any section after Section 5.
- If content is partial or a fragment, extract every technical detail available and present it confidently.

---

PAPER CONTENT:
{context}

---

Your analysis MUST cover ALL five sections below:

## 1. WHAT IS THIS PAPER ABOUT?
- What problem does it solve?
- What is the proposed solution, method, or algorithm?
- What domain is it in? (e.g. NLP, computer vision, reinforcement learning, graph networks)

## 2. WHAT DID THE AUTHORS IMPLEMENT?
- List every major component they built
- Describe the full data flow from input to output
- What are the exact inputs and outputs of the system?

## 3. CORE ALGORITHMS & EQUATIONS
- List every key algorithm step by step
- Include every mathematical operation with variable names exactly as used in the paper
- Describe any special data structures used

## 4. HOW WOULD YOU REPLICATE THIS IN PYTHON?
- What Python libraries would best implement each component?
- What are the natural module boundaries (one module per major component)?
- How do the components depend on each other?

## 5. IMPLEMENTATION RECIPE
Write a numbered step-by-step guide a developer would follow:
Step 1: ...
Step 2: ...
Step 3: ...
Step 4: ...
Step 5: ...

Be specific and technical. Use the paper's own terminology and variable names.

STOP after Step 5. Do not write anything after the last implementation step.
""",
        expected_output="""A detailed technical analysis document covering all five sections:
- Paper summary and domain (Section 1)
- All components and their data flow (Section 2)
- Core algorithms with steps and equations (Section 3)
- Recommended Python libraries and module design (Section 4)
- Step-by-step implementation recipe (Section 5)""",
        agent=agent
    )


def make_architecture_task(analysis: str, agent) -> Task:
    return Task(
        description=f"""Based on this technical analysis of a research paper, design a complete Python project structure.

PAPER ANALYSIS:
{analysis}

Return ONLY the JSON object below. Fill in every field based on the paper analysis above.
Every string value must be on ONE single line — no newline characters inside any string value.

{{
  "project_name": "short-hyphenated-name-from-paper",
  "description": "One sentence describing what this project implements",
  "folders": ["src"],
  "files": [
    {{
      "filename": "src/component.py",
      "purpose": "One sentence: what this file does and which paper component it implements",
      "dependencies": ["torch"],
      "classes": ["ExactClassName"],
      "functions": ["exact_function_name"],
      "logic_summary": "One line: the core computation this file performs",
      "class_details": {{
        "ExactClassName": "What this class represents, its constructor arguments, and its forward/call method behavior"
      }},
      "function_details": {{
        "exact_function_name": "What this function computes, its arguments, return value, and which paper equation it implements"
      }}
    }}
  ],
  "dependencies": ["torch", "numpy"],
  "main_file": "src/main.py",
  "readme_content": "Plain text description of the project and how to run it. No backticks."
}}

RULES — EVERY ONE IS MANDATORY:
1. Return ONLY the JSON object — nothing before it, nothing after it, no markdown fences
2. Every string value must be ONE line — no newline characters (\\n) inside any string
3. "main_file" must always be exactly "src/main.py"
4. The root "dependencies" list must include ALL pip packages from ALL files combined — never empty
5. ALWAYS include "src/main.py" as the LAST entry in the "files" list
6. "src/main.py" must have: classes[] = [], functions[] = ["main"], and its function_details["main"] must name every class it will import from other files
7. class_details keys must exactly match the entries in that file's classes[] list
8. function_details keys must exactly match the entries in that file's functions[] list
9. dependencies per file: pip package names only — no version numbers (write "torch" not "torch==2.0")
10. readme_content: one line of plain text — no commas, no newlines
11. Derive ALL filenames, class names, and function names from the paper analysis — never use generic names like "Model" or "run" unless the paper uses them
12. If a file only contains functions, its classes[] must be [] and class_details must be {{}}
13. Do NOT include algorithm_steps — omit that field entirely from every file
""",
        expected_output="""A valid JSON project plan with no text outside the JSON.
The last file is always src/main.py.
Root dependencies includes all pip packages from all files.
main_file is exactly "src/main.py".""",
        agent=agent
    )


def make_code_task(file_plan: dict, rag_context: str, agent) -> Task:
    classes_list = ", ".join(file_plan.get("classes", [])) or "None"
    functions_list = ", ".join(file_plan.get("functions", [])) or "None"

    class_details_str = "\n".join(
        f"  {cls}: {desc}"
        for cls, desc in file_plan.get("class_details", {}).items()
    ) or "  (none — implement as described in logic_summary)"

    func_details_str = "\n".join(
        f"  {fn}: {desc}"
        for fn, desc in file_plan.get("function_details", {}).items()
    ) or "  (none — implement as described in logic_summary)"

    return Task(
        description=f"""Write a complete, working Python file.

FILE TO CREATE: {file_plan['filename']}
PURPOSE: {file_plan['purpose']}
CORE LOGIC: {file_plan['logic_summary']}

EXACT NAMES YOU MUST USE — DO NOT CHANGE OR INVENT NEW ONES:
- Classes to implement: {classes_list}
- Functions to implement: {functions_list}

REQUIRED IMPORTS (pip packages available):
{file_plan.get('dependencies', [])}

CLASS SPECIFICATIONS:
{class_details_str}

FUNCTION SPECIFICATIONS:
{func_details_str}

PAPER TECHNICAL CONTEXT (use this for equations, variable names, hyperparameters):
{rag_context}

CODING REQUIREMENTS:
1. Start the file with all import statements — nothing before the imports
2. Implement every class listed: {classes_list}
3. Implement every function listed: {functions_list}
4. Add type hints to every function signature
5. Every method and function must contain real, working logic — no `pass`, no `...`, no placeholders
6. Add inline comments referencing paper equations where applicable, e.g. # Paper Eq.3: softmax(QK^T / sqrt(d_k))
7. Use exact variable names from the paper where applicable
8. Do NOT write any explanation, note, or markdown after the last line of code
9. The file must be valid Python that can be imported without errors

Output ONLY the Python source code. Nothing else.
""",
        expected_output="""Complete valid Python source code.
Starts with import statements.
Ends with the last line of executable or class/function code.
No markdown, no notes, no explanations outside of inline code comments.""",
        agent=agent
    )


def make_review_task(filename: str, code: str, agent) -> Task:
    return Task(
        description=f"""Review this Python implementation and return a JSON report.

FILE: {filename}

CODE:
{code}

Check ALL of the following:
1. Does the file start with import statements (no prose before imports)?
2. Are all classes implemented with real logic (not stubs, not just `pass`)?
3. Are all functions implemented with real logic?
4. Is the code syntactically valid Python?
5. Are there any trailing notes, markdown, or prose after the last line of code?

Return ONLY this JSON object — nothing before it, nothing after it:
{{
  "status": "ok",
  "issues": ["describe each issue found, or leave empty list if none"],
  "suggestion": "one line fix suggestion if issues found, empty string if status is ok"
}}
""",
        expected_output="A JSON object with keys: status, issues, suggestion. No other text.",
        agent=agent
    )
