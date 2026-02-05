"""Workflow creation tool for krolik."""

import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult


class CreateWorkflowTool(Tool):
    """
    Tool for creating new workflow files in .windsurf/workflows/.
    
    Workflows are markdown files that define reusable procedures
    for the agent to follow.
    """
    
    name = "create_workflow"
    description = """Create a new workflow file in .windsurf/workflows/.

Workflows are reusable procedures that define step-by-step instructions
for common tasks. They follow a YAML frontmatter + markdown format.

Use this when:
- The user wants to automate a recurring task
- You notice a pattern that should be standardized
- Creating reusable procedures for specific domains

Example: create_workflow(name="deploy-check", description="Pre-deployment checks")
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Workflow name (kebab-case, no spaces)"
            },
            "description": {
                "type": "string",
                "description": "Short description of what this workflow does"
            },
            "steps": {
                "type": "array",
                "description": "List of workflow steps (optional, can be added later)",
                "items": {
                    "type": "string"
                },
                "default": []
            },
            "content": {
                "type": "string",
                "description": "Full markdown content (optional, overrides steps)",
                "default": None
            }
        },
        "required": ["name", "description"]
    }
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workflows_dir = workspace / ".windsurf" / "workflows"
    
    async def execute(
        self,
        name: str,
        description: str,
        steps: list[str] | None = None,
        content: str | None = None
    ) -> ToolResult:
        """Create a new workflow file."""
        try:
            # Validate name
            if not self._validate_name(name):
                return ToolResult(
                    success=False,
                    error=f"Invalid workflow name '{name}'. Use kebab-case (letters, numbers, hyphens only)."
                )
            
            # Ensure directory exists
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Build file path
            workflow_path = self.workflows_dir / f"{name}.md"
            
            # Check if exists
            if workflow_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Workflow '{name}' already exists at {workflow_path}"
                )
            
            # Build content
            if content:
                workflow_content = self._ensure_frontmatter(content, name, description)
            else:
                workflow_content = self._build_workflow(name, description, steps or [])
            
            # Write file
            workflow_path.write_text(workflow_content, encoding="utf-8")
            
            logger.info(f"Created workflow: {workflow_path}")
            
            return ToolResult(
                success=True,
                output=f"✅ Created workflow '{name}' at {workflow_path}\n\n{description}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _validate_name(self, name: str) -> bool:
        """Validate workflow name (kebab-case)."""
        return bool(re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name))
    
    def _build_workflow(self, name: str, description: str, steps: list[str]) -> str:
        """Build workflow markdown content."""
        lines = [
            "---",
            f"description: {description}",
            "---",
            "",
            f"# {name.replace('-', ' ').title()}",
            "",
            f"{description}",
            "",
        ]
        
        if steps:
            lines.extend([
                "## Steps",
                ""
            ])
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")
        else:
            lines.extend([
                "## Steps",
                "",
                "1. // TODO: Add step 1",
                "2. // TODO: Add step 2",
                "",
            ])
        
        lines.extend([
            "## Usage",
            "",
            f"Run this workflow: `krolik workflow run {name}`",
            ""
        ])
        
        return "\n".join(lines)
    
    def _ensure_frontmatter(self, content: str, name: str, description: str) -> str:
        """Ensure content has proper frontmatter."""
        if content.startswith("---"):
            # Already has frontmatter
            return content
        
        # Add frontmatter
        frontmatter = f"""---
description: {description}
---

"""
        return frontmatter + content


class ListWorkflowsTool(Tool):
    """Tool to list available workflows."""
    
    name = "list_workflows"
    description = """List all available workflows in .windsurf/workflows/.

Shows workflow names, descriptions, and file paths.
Use this to discover available workflows before running them.
"""
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workflows_dir = workspace / ".windsurf" / "workflows"
    
    async def execute(self) -> ToolResult:
        """List all workflows."""
        try:
            if not self.workflows_dir.exists():
                return ToolResult(
                    success=True,
                    output="No workflows directory found. Create one with 'create_workflow'."
                )
            
            workflows = []
            for workflow_file in self.workflows_dir.glob("*.md"):
                name = workflow_file.stem
                desc = self._extract_description(workflow_file)
                workflows.append((name, desc))
            
            if not workflows:
                return ToolResult(
                    success=True,
                    output="No workflows found. Create one with 'create_workflow'."
                )
            
            lines = ["Available workflows:", ""]
            for name, desc in sorted(workflows):
                lines.append(f"• {name}: {desc}")
            
            return ToolResult(
                success=True,
                output="\n".join(lines)
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _extract_description(self, workflow_file: Path) -> str:
        """Extract description from workflow frontmatter."""
        try:
            content = workflow_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                match = re.search(r'^---\n.*?description:\s*(.+?)\n', content, re.DOTALL)
                if match:
                    return match.group(1).strip()
            return "No description"
        except Exception:
            return "Error reading file"


class RunWorkflowTool(Tool):
    """Tool to execute a workflow."""
    
    name = "run_workflow"
    description = """Read and execute a workflow from .windsurf/workflows/.

Loads the workflow file and follows its defined steps.
Use this to execute reusable procedures.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Workflow name (without .md extension)"
            }
        },
        "required": ["name"]
    }
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workflows_dir = workspace / ".windsurf" / "workflows"
    
    async def execute(self, name: str) -> ToolResult:
        """Load and execute a workflow."""
        try:
            workflow_path = self.workflows_dir / f"{name}.md"
            
            if not workflow_path.exists():
                return ToolResult(
                    success=False,
                    error=f"Workflow '{name}' not found at {workflow_path}"
                )
            
            content = workflow_path.read_text(encoding="utf-8")
            
            # Parse workflow
            workflow_data = self._parse_workflow(content)
            
            return ToolResult(
                success=True,
                output=f"📋 Workflow: {workflow_data['title']}\n\n{workflow_data['description']}\n\n## Steps\n\n{self._format_steps(workflow_data['steps'])}"
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _parse_workflow(self, content: str) -> dict[str, Any]:
        """Parse workflow markdown content."""
        # Extract frontmatter
        desc = "No description"
        if content.startswith("---"):
            match = re.search(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if match:
                frontmatter = match.group(1)
                desc_match = re.search(r'^description:\s*(.+)$', frontmatter, re.MULTILINE)
                if desc_match:
                    desc = desc_match.group(1).strip()
                # Remove frontmatter from content
                content = content[match.end():]
        
        # Extract title (first H1)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled Workflow"
        
        # Extract steps
        steps = []
        steps_match = re.search(r'##\s*Steps\s*\n(.*?)(?=##|$)', content, re.DOTALL | re.IGNORECASE)
        if steps_match:
            steps_text = steps_match.group(1).strip()
            for line in steps_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('- ') or re.match(r'^\d+\.\s', line)):
                    # Remove list markers
                    step = re.sub(r'^[\-\*\d\.]\s*', '', line)
                    steps.append(step)
        
        return {
            "title": title,
            "description": desc,
            "steps": steps
        }
    
    def _format_steps(self, steps: list[str]) -> str:
        """Format steps for display."""
        if not steps:
            return "(No steps defined)"
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
