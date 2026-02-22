from typing import Dict, List
from pydantic import BaseModel, Field


class FileSpec(BaseModel):
    filename: str = Field(..., description="e.g. models/attention.py")
    purpose: str = Field(..., description="One sentence — what this file does")
    dependencies: List[str] = Field(default=[])
    classes: List[str] = Field(default=[])
    functions: List[str] = Field(default=[])
    logic_summary: str = Field(default="")
    class_details: Dict[str, str] = Field(default={})
    function_details: Dict[str, str] = Field(default={})
    algorithm_steps: List[str] = Field(default=[])


class ProjectPlan(BaseModel):
    project_name: str
    description: str
    folders: List[str]
    files: List[FileSpec] = Field(..., min_length=1)
    dependencies: List[str] = Field(default=[])
    main_file: str
    readme_content: str = Field(default="Research paper implementation.")
