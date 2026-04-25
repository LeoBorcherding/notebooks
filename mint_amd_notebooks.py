#!/usr/bin/env python3
"""
mint_amd_notebooks.py
---------------------
Generates AMD- prefixed notebooks in nb/ from original_template/*.ipynb.
Existing Colab and Kaggle notebooks are left completely untouched.

Run from the repo root:
    python mint_amd_notebooks.py [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
from glob import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# AMD-only install templates (no COLAB_ / is_fresh branching)
# ROCm/AMD: torch is pre-installed as a ROCm build — skip torch/triton.
# ---------------------------------------------------------------------------

INSTALL_AMD_BASE = """\
%%capture
import os, importlib.util, subprocess, sys

def _pip(*packages):
    try:
        if subprocess.run(["uv", "--version"], capture_output=True).returncode == 0:
            cmd = ["uv", "pip", "install", "--system", "-qqq"]
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        cmd = [sys.executable, "-m", "pip", "install", "-qqq"]
    subprocess.run(cmd + list(packages), check=False)

import socket
try:
    socket.getaddrinfo("huggingface.co", 443, socket.AF_INET)
except socket.gaierror:
    with open("/etc/resolv.conf", "a") as _f:
        _f.write("nameserver 8.8.8.8\\nnameserver 8.8.4.4\\n")
# ROCm/AMD: torch already installed as ROCm build \u2014 skip torch/triton, use [amd] extra
try: import numpy; _np = f"numpy=={numpy.__version__}"
except: _np = "numpy"
try: import PIL; _pil = f"pillow=={PIL.__version__}"
except: _pil = "pillow"
_pip(_np, _pil, "bitsandbytes", "cut-cross-entropy", "torchao")
_pip("--no-deps",
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
    "unsloth[amd] @ git+https://github.com/unslothai/unsloth",
)
_pip("--upgrade", "--no-deps",
    "transformers>=5.0.0", "tokenizers", "huggingface_hub>=1.5.0",
    "trl>=0.24.0", "unsloth", "unsloth_zoo",
)
"""

INSTALL_AMD_GRPO = """\
%%capture
import os, importlib.util, subprocess, sys

def _pip(*packages):
    try:
        if subprocess.run(["uv", "--version"], capture_output=True).returncode == 0:
            cmd = ["uv", "pip", "install", "--system", "-qqq"]
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        cmd = [sys.executable, "-m", "pip", "install", "-qqq"]
    subprocess.run(cmd + list(packages), check=False)

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import socket
try:
    socket.getaddrinfo("huggingface.co", 443, socket.AF_INET)
except socket.gaierror:
    with open("/etc/resolv.conf", "a") as _f:
        _f.write("nameserver 8.8.8.8\\nnameserver 8.8.4.4\\n")
# ROCm/AMD: torch already installed as ROCm build \u2014 skip torch/triton, use [amd] extra
try: import numpy; _np = f"numpy=={numpy.__version__}"
except: _np = "numpy"
_pip(_np, "bitsandbytes", "cut-cross-entropy", "torchao")
_pip("--no-deps",
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
    "unsloth[amd] @ git+https://github.com/unslothai/unsloth",
)
_pip("--upgrade", "--no-deps",
    "transformers>=5.0.0", "tokenizers", "huggingface_hub>=1.5.0",
    "trl>=0.24.0", "unsloth", "unsloth_zoo",
)
"""

INSTALL_AMD_GEMMA4 = """\
%%capture
import os, importlib.util, subprocess, sys

def _pip(*packages):
    try:
        if subprocess.run(["uv", "--version"], capture_output=True).returncode == 0:
            cmd = ["uv", "pip", "install", "--system", "-qqq"]
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        cmd = [sys.executable, "-m", "pip", "install", "-qqq"]
    subprocess.run(cmd + list(packages), check=False)

import socket
try:
    socket.getaddrinfo("huggingface.co", 443, socket.AF_INET)
except socket.gaierror:
    with open("/etc/resolv.conf", "a") as _f:
        _f.write("nameserver 8.8.8.8\\nnameserver 8.8.4.4\\n")
# ROCm/AMD: torch already installed as ROCm build \u2014 skip torch/triton, use [amd] extra
try: import numpy; _np = f"numpy=={numpy.__version__}"
except: _np = "numpy"
try: import PIL; _pil = f"pillow=={PIL.__version__}"
except: _pil = "pillow"
_pip(_np, _pil, "bitsandbytes", "cut-cross-entropy", "torchao")
_pip("--no-deps",
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
    "unsloth[amd] @ git+https://github.com/unslothai/unsloth",
)
# Gemma 4 requires transformers >= 5.5.0
_pip("--upgrade", "--no-deps",
    "transformers>=5.5.0", "tokenizers", "huggingface_hub>=1.5.0",
    "trl>=0.28.0", "unsloth", "unsloth_zoo",
)
"""

# AMD Dev Cloud announcement prefix (replaces Colab T4 wording)
AMD_ANNOUNCEMENT_SNIPPET = "on **AMD Dev Cloud**!"
COLAB_ANNOUNCEMENT_SNIPPET = "on a **free** Tesla T4 Google Colab instance!"
COLAB_RUN_PHRASE = 'press "*Runtime*" and press "*Run all*"'
AMD_RUN_PHRASE = 'press "*Run*" and press "*Run All*"'

# ---------------------------------------------------------------------------
# Install cell detection (same patterns as update_all_notebooks.py)
# ---------------------------------------------------------------------------

_INSTALL_PATTERNS = [
    r"pip install.*unsloth",
    r"uv pip install",
    r"COLAB_.*pip install",
    r"find_spec.*torch.*pip install",
    r"pip install unsloth",
    r"def _pip\(",
]


def _is_install_cell(source: str) -> bool:
    return any(re.search(p, source, re.DOTALL) for p in _INSTALL_PATTERNS)


def _pick_template(name_lower: str) -> str:
    if "gemma4" in name_lower or "gemma_4" in name_lower:
        return INSTALL_AMD_GEMMA4
    if any(x in name_lower for x in ["grpo", "reinforcement", "rl_", "synthetic-data",
                                      "minesweeper", "2048", "sudoku"]):
        return INSTALL_AMD_GRPO
    return INSTALL_AMD_BASE


def _patch_announcement(source: str) -> str:
    """Replace Colab T4 wording with AMD Dev Cloud wording in a markdown cell."""
    source = source.replace(COLAB_ANNOUNCEMENT_SNIPPET, AMD_ANNOUNCEMENT_SNIPPET)
    source = source.replace(COLAB_RUN_PHRASE, AMD_RUN_PHRASE)
    return source


def _source_str(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _find_install_cell_index(cells: list) -> int:
    """
    Return the index of the install code cell.
    Strategy 1: code cell immediately after '### Installation' markdown.
    Strategy 2: first code cell whose source matches known install patterns.
    Returns -1 if not found.
    """
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            if "### Installation" in _source_str(cell):
                # Look for the next code cell
                for j in range(i + 1, len(cells)):
                    if cells[j].get("cell_type") == "code":
                        return j
                    if cells[j].get("cell_type") == "markdown":
                        break  # hit another heading, stop
    # Fallback: content-based search
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and _is_install_cell(_source_str(cell)):
            return i
    return -1


def mint_amd_notebook(template_path: Path, dest_path: Path, dry_run: bool = False) -> bool:
    """
    Copy template_path -> dest_path and apply AMD-only install + announcement.
    Returns True if the file was written (or would be written in dry_run).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  SKIP (invalid JSON): {e}")
            return False

    name_lower = template_path.name.lower()
    install_template = _pick_template(name_lower)
    modified = False

    # Patch first markdown cell (announcement)
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            src = _source_str(cell)
            patched = _patch_announcement(src)
            if patched != src:
                nb["cells"][i]["source"] = [patched]
                modified = True
            break  # only patch the first markdown cell

    # Find and patch install cell
    cells = nb.get("cells", [])
    install_idx = _find_install_cell_index(cells)

    if install_idx == -1:
        print(f"  SKIP (no install cell found)")
        return False

    is_grpo = any(x in name_lower for x in ["grpo", "reinforcement", "rl_", "sudoku", "2048", "minesweeper"])
    extra_cell_deleted = False

    current_src = _source_str(cells[install_idx])
    if current_src.strip() == install_template.strip():
        print(f"  Already up to date")
        # Still write the file if announcement was patched
        if not modified:
            return False
    else:
        nb["cells"][install_idx]["source"] = [install_template]
        nb["cells"][install_idx]["outputs"] = []
        nb["cells"][install_idx]["execution_count"] = None
        modified = True

    # GRPO notebooks have a second install cell (extra Colab cell).
    # For AMD we delete it — single self-contained install is enough.
    if is_grpo and install_idx + 1 < len(nb["cells"]):
        next_cell = nb["cells"][install_idx + 1]
        if next_cell.get("cell_type") == "code":
            next_src = _source_str(next_cell)
            if _is_install_cell(next_src) or "uv pip install" in next_src or "Placeholder" in next_src:
                del nb["cells"][install_idx + 1]
                extra_cell_deleted = True

    if not modified:
        print(f"  No changes needed")
        return False

    if dry_run:
        tname = ("GEMMA4" if "gemma4" in name_lower else
                 "GRPO" if is_grpo else "BASE")
        print(f"  Would write AMD-{template_path.name} [{tname}]"
              + (" + deleted extra GRPO cell" if extra_cell_deleted else ""))
        return True

    # Preserve indent style of the existing file if it exists
    indent = 1
    if dest_path.exists():
        try:
            raw = dest_path.read_text(encoding="utf-8")
            indent = 2 if '"cells": [' in raw[:500] else 1
        except Exception:
            pass

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=indent, ensure_ascii=False)
        f.write("\n")

    tname = ("GEMMA4" if "gemma4" in name_lower else
             "GRPO" if is_grpo else "BASE")
    print(f"  Written AMD-{template_path.name} [{tname}]"
          + (" + deleted extra GRPO cell" if extra_cell_deleted else ""))
    return True


def main():
    parser = argparse.ArgumentParser(description="Mint AMD- notebooks from original_template/")
    parser.add_argument("--template-dir", default="original_template",
                        help="Source template directory (default: original_template)")
    parser.add_argument("--nb-dir", default="nb",
                        help="Output notebook directory (default: nb)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without writing")
    args = parser.parse_args()

    template_dir = Path(args.template_dir)
    nb_dir = Path(args.nb_dir)

    if not template_dir.exists():
        print(f"ERROR: template dir '{template_dir}' does not exist")
        raise SystemExit(1)

    nb_dir.mkdir(parents=True, exist_ok=True)

    templates = sorted(template_dir.glob("*.ipynb"))
    print(f"Found {len(templates)} templates in '{template_dir}'")
    print(f"Output dir: '{nb_dir}' (dry_run={args.dry_run})\n")

    written = 0
    skipped = 0

    for tpl in templates:
        amd_name = "AMD-" + tpl.name
        dest = nb_dir / amd_name
        print(f"Processing: {tpl.name}")
        ok = mint_amd_notebook(tpl, dest, dry_run=args.dry_run)
        if ok:
            written += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    action = "Would write" if args.dry_run else "Written"
    print(f"{action}: {written}  |  Skipped: {skipped}")
    if not args.dry_run and written:
        print(f"\nDone! {written} AMD notebooks created in '{nb_dir}/'")
        print("Commit with:  git add nb/AMD-*.ipynb && git commit -m 'feat: add AMD Dev Cloud notebooks'")


if __name__ == "__main__":
    main()
