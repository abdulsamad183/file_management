from langchain_core.tools import tool
from pathlib import Path
import shutil

# --------------------
# Configuration
# --------------------
ROOT = Path.cwd() / "workspace"
ROOT.mkdir(exist_ok=True)
ALLOWED_ROOT = ROOT.resolve()


# --------------------
# Helper: sandbox-safe path join
# --------------------
def in_sandbox_path(rel_path: str) -> Path:
    """
    Resolve a path *relative to the sandbox* and ensure it stays inside.
    """
    candidate = (ALLOWED_ROOT / rel_path).resolve()
    if not str(candidate).startswith(str(ALLOWED_ROOT)):
        raise ValueError("Path escapes the sandbox root.")
    return candidate

# --------------------
# Tools: filesystem ops (exposed to the LLM)
# --------------------
@tool
def create_folder(path: str) -> str:
    """Create a folder inside the workspace. Args: path (relative)."""
    p = in_sandbox_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Folder created: {p.relative_to(ALLOWED_ROOT)}"

@tool
def list_folder(path: str = ".") -> str:
    """List files/folders inside workspace (relative path)."""
    p = in_sandbox_path(path)
    if not p.exists():
        return f"Path not found: {p.relative_to(ALLOWED_ROOT)}"
    items = []
    for child in sorted(p.iterdir()):
        t = "DIR" if child.is_dir() else "FILE"
        items.append(f"{t}\t{child.name}")
    return "\n".join(items) or "(empty)"

@tool
def create_file(path: str, content: str = "") -> str:
    """Create a file (relative path) with optional content."""
    p = in_sandbox_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File created: {p.relative_to(ALLOWED_ROOT)} (bytes={p.stat().st_size})"

@tool
def edit_file(path: str, new_content: str) -> str:
    """Replace the contents of a file with new_content."""
    p = in_sandbox_path(path)
    if not p.exists() or not p.is_file():
        return f"File not found: {p.relative_to(ALLOWED_ROOT)}"
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_content)
    return f"File edited: {p.relative_to(ALLOWED_ROOT)} (bytes={p.stat().st_size})"

@tool
def rename(path_from: str, path_to: str) -> str:
    """Rename/move a file or folder inside the workspace."""
    src = in_sandbox_path(path_from)
    dst = in_sandbox_path(path_to)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return f"Source not found: {src.relative_to(ALLOWED_ROOT)}"
    shutil.move(str(src), str(dst))
    return f"Moved {path_from} -> {path_to}"

@tool
def delete(path: str) -> str:
    """Delete a file or folder (recursive)."""
    p = in_sandbox_path(path)
    if not p.exists():
        return f"Path not found: {p.relative_to(ALLOWED_ROOT)}"
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return f"Deleted: {p.relative_to(ALLOWED_ROOT)}"
