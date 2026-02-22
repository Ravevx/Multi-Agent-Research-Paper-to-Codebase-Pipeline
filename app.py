import streamlit as st
import os
import io
import zipfile
from pathlib import Path
from crew_runner import load_paper_chunks, load_paper_rag, analyse_paper, generate_plan, generate_code, save_project
from config import PROJECT_OUTPUT_DIR
from project_planner import ProjectPlan


st.set_page_config(page_title="Research → Code Agent", layout="wide")
st.title("Research Paper → Complete Codebase")
st.caption("Upload any AI/ML/Data Science paper → AI agents analyse it and generate a full Python project")


# Helper
def generate_zip(project_dir: str) -> bytes:
    project_path = Path(project_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in project_path.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(project_path))
    buffer.seek(0)
    return buffer.getvalue()


# Step 1: Paper Input
st.subheader("📥 Step 1: Upload Paper")
col_upload, col_arxiv = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload research paper PDF", type="pdf")

with col_arxiv:
    arxiv_id = st.text_input("Or enter arXiv ID", placeholder="e.g. 1706.03762")

paper_path = None
if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    paper_path = f"temp/{uploaded_file.name}"
    with open(paper_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved: {paper_path}")
elif arxiv_id.strip():
    paper_path = arxiv_id.strip()
    st.success(f"Using arXiv ID: {paper_path}")


# Step 2: Analyse Button
if paper_path:
    if st.button("🔍 Analyse Paper & Generate Plan", type="primary"):
        progress = st.progress(0, text="📄 Loading paper chunks...")
        status_box = st.empty()

        try:
            # Load chunks
            chunks = load_paper_chunks(paper_path)

            # ── BUILD RAG ONCE ──────────────────────────────────────
            progress.progress(15, text="🔨 Building RAG index...")
            status_box.info("🔨 Building RAG index from paper...")
            rag_store = load_paper_rag(chunks)
            # ────────────────────────────────────────────────────────

            progress.progress(25, text="🔬 Analyst agent reading paper...")
            status_box.info("🔬 **Analyst** is deeply reading the paper — understanding algorithms, data flow, equations...")

            # Step 1: Analyst understands paper
            analysis = analyse_paper(rag_store)
            progress.progress(60, text="🏗️ Architect agent designing project structure...")
            status_box.info("🏗️ **Architect** is designing the Python project structure based on the analysis...")

            # Step 2: Architect designs plan
            plan = generate_plan(analysis)
            progress.progress(100, text="Plan ready!")
            status_box.success("Plan generated! Review it below.")

            # Save to session
            st.session_state["plan"] = plan
            st.session_state["analysis"] = analysis
            st.session_state["rag_store"] = rag_store
            st.session_state["plan_ready"] = True
            st.session_state["code_done"] = False
            st.rerun()

        except Exception as e:
            progress.empty()
            status_box.empty()
            st.error(f"❌ Error during analysis: {str(e)}")
            raise e


# Step 3: Review Plan
if st.session_state.get("plan_ready") and not st.session_state.get("code_done"):
    plan: ProjectPlan = st.session_state["plan"]
    analysis: str = st.session_state.get("analysis", "")

    st.divider()
    st.subheader("📋 Step 2: Review Generated Plan")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**🏷️ Project:** `{plan.project_name}`")
        st.markdown(f"**📝 Description:** {plan.description}")
        st.markdown("**📁 Folders:** " + "  ".join([f"`{f}/`" for f in plan.folders]))
    with col2:
        st.markdown("**📦 Dependencies:**")
        st.code("\n".join(plan.dependencies) if plan.dependencies else "(auto-detected)")
        st.markdown(f"**▶️ Entry point:** `{plan.main_file}`")

    if analysis:
        with st.expander("🔬 View Analyst's Paper Understanding", expanded=False):
            st.markdown(analysis)

    st.divider()

    st.markdown(f"### 📄 {len(plan.files)} Files to Generate")
    for i, f in enumerate(plan.files, 1):
        with st.expander(f"**{i}. `{f.filename}`** — {f.purpose}", expanded=True):
            st.markdown(f"**📌 Purpose:** {f.purpose}")
            if f.logic_summary:
                st.info(f"🧮 **Core Logic:** {f.logic_summary}")

            if f.algorithm_steps:
                st.markdown("**📐 Algorithm Steps:**")
                for step in f.algorithm_steps:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{step}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🧱 Classes:**")
                if f.classes:
                    for cls in f.classes:
                        detail = f.class_details.get(cls, "") if f.class_details else ""
                        if detail:
                            st.markdown(f"- **`{cls}`**  \n  ↳ {detail}")
                        else:
                            st.markdown(f"- `{cls}`")
                else:
                    st.markdown("*(none)*")

            with col2:
                st.markdown("**⚙️ Functions:**")
                if f.functions:
                    for fn in f.functions:
                        detail = f.function_details.get(fn, "") if f.function_details else ""
                        if detail:
                            st.markdown(f"- **`{fn}()`**  \n  ↳ {detail}")
                        else:
                            st.markdown(f"- `{fn}()`")
                else:
                    st.markdown("*(none)*")

            if f.dependencies:
                st.markdown(f"**🔗 Imports:** `{'`, `'.join(f.dependencies)}`")

    st.divider()
    st.subheader("✋ Step 3: Your Decision")
    st.warning(
        "⚠️ Review every file carefully above. "
        "Code generation runs one file at a time and may take 5–15 minutes depending on file count."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve — Generate Code", type="primary", use_container_width=True):
            progress = st.progress(0, text="🤖 Starting code generation...")
            status_box = st.empty()

            try:
                analysis = st.session_state["analysis"]
                plan = st.session_state["plan"]
                rag_store = st.session_state["rag_store"]   # ── RETRIEVE RAG FROM SESSION
                total = len(plan.files)

                status_box.info(f"💻 **Coder** writing {total} files...")

                files = generate_code(plan, analysis, rag_store)   # ── PASS RAG TO CODER

                progress.progress(90, text="💾 Saving project to disk...")
                status_box.info("💾 Saving files to disk...")

                project_dir = save_project(plan, files)

                progress.progress(100, text="✅ Done!")
                status_box.success("✅ All files generated!")

                st.session_state["files_written"] = files
                st.session_state["project_dir"] = project_dir
                st.session_state["code_done"] = True
                st.session_state["plan_ready"] = False
                st.rerun()

            except Exception as e:
                progress.empty()
                status_box.empty()
                st.error(f"❌ Error during code generation: {str(e)}")
                raise e

    with col2:
        if st.button("❌ Reject — Start Over", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# Step 4: Results
if st.session_state.get("code_done"):
    files_written: dict = st.session_state.get("files_written", {})
    plan: ProjectPlan = st.session_state.get("plan")
    project_dir: str = st.session_state.get("project_dir", "")

    st.divider()
    st.subheader("🎉 Step 4: Generated Codebase")

    if files_written:
        total = len(files_written)
        ok_files = sum(1 for c in files_written.values() if not c.startswith("# Syntax error"))
        err_files = total - ok_files

        m1, m2, m3 = st.columns(3)
        m1.metric("📄 Total Files", total)
        m2.metric("✅ Clean Files", ok_files)
        m3.metric("⚠️ Files with Errors", err_files)

        st.success(f"✅ Project `{plan.project_name}` generated!")

        st.markdown("### 📁 Files")
        for filename, code in files_written.items():
            lines = len(code.splitlines())
            has_error = code.startswith("# Syntax error")
            icon = "⚠️" if has_error else "✅"
            label = f"{icon} `{filename}` — {lines} lines"
            with st.expander(label, expanded=False):
                st.code(code, language="python")

        st.divider()
        if Path(project_dir).exists():
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download Full Project as ZIP",
                    data=generate_zip(project_dir),
                    file_name=f"{plan.project_name}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            with col2:
                st.info(f"📂 Saved locally at: `{project_dir}`")

    else:
        st.warning("⚠️ No files were generated. Check the terminal logs for errors.")

    st.divider()
    if st.button("🔁 Start Over with New Paper", use_container_width=True):
        st.session_state.clear()
        st.rerun()
