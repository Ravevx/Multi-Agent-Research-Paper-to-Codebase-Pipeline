from fastapi import FastAPI, UploadFile, Form
from code_graph import app as graph_app

app = FastAPI()

@app.post("/start-research")
async def start_research(paper: UploadFile):
    thread_id = f"research-{hash(paper.filename)}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run until human review
    result = graph_app.invoke({"paper_path": paper.filename}, config)
    return {
        "thread_id": thread_id,
        "status": "waiting_human_approval",
        "plan": result["project_plan"].model_dump()
    }

@app.post("/approve-plan/{thread_id}")
async def approve_plan(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    # Resume from human review
    result = graph_app.invoke({"human_feedback": "approved"}, config)
    return {"status": "code_generated", "files": result["files_written"]}
