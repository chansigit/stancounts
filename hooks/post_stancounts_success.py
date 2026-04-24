#!/usr/bin/env python3
"""
stancounts PostToolUse hook — family handoff to QC stage.

stancounts ships as a Python library (no CLI), so this hook is best-effort:
it triggers when a Bash ``python …`` invocation mentions ``stancounts`` as
a module/import, e.g.::

    python -c "from stancounts import reverse_log1p_anndata; ..."
    python some_script.py            # NOT detectable — script may import internally
    python -m stancounts.xxx         # detectable (though stancounts has no CLI today)

When the Python process exits 0, inject a handoff hint: the next step in
the stan* family pipeline is the QC stage (eca-curation's 03_qc / future
stanqc module; presently implemented on top of scanpy).

Design rules:
- Never block. This is PostToolUse — the tool already executed.
- No-op unless the Bash command is a ``python …`` that explicitly mentions
  ``stancounts`` (prevents false positives like ``pip install stancounts``).
- No-op on non-zero exit, parse error, or missing field.
- Exit 0 in all cases. Feedback is delivered via JSON stdout
  (``hookSpecificOutput.additionalContext``), with stderr as a
  best-effort fallback for older runtimes.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Optional

# Match: `python[3] …  stancounts …` within a single shell command.
# Anchoring to `python` prevents false positives on `cat stancounts_notes.md`,
# `conda install -c bioconda stancounts`, etc. The explicit package-manager
# filter below covers `python -m pip install stancounts` and similar forms
# where `python` itself is the package-manager launcher.
_STANCOUNTS_CMD = re.compile(
    r"(?:^|[\s;|&\n])python\d?\s[^\n]*?\bstancounts\b"
)

_PKG_MGR_CMD = re.compile(
    r"\b(?:pip|pip3|conda|mamba|micromamba|uv|poetry|pipx)"
    r"\s+(?:install|uninstall|add|remove|show|list|search|info|update|upgrade|sync)\b"
)

_HANDOFF = (
    "✅ stancounts 执行成功(已反向 log1p,恢复 integer counts)。\n"
    "\n"
    "家族流水线中 stancounts 的下游是 **QC 阶段**。QC 目前由 eca-curation 的\n"
    "iterative stage 03_qc 驱动,由 qc-iterator subagent 按 rubric 自主迭代参数。\n"
    "(独立的 stanqc 模块在规划中;在 stanqc 成熟前,QC 实现基于 scanpy。\n"
    "stanqc 上线后是同一路径的平滑替换,不影响现在的工作流。)\n"
    "\n"
    "建议的下一步:\n"
    "  • 如果当前在 eca-curation pipeline session 里:运行\n"
    "      /eca-run <dataset>\n"
    "    由 qc-iterator subagent 接管 QC 迭代(读 rubric → 决定参数 → 跑 stage\n"
    "    script → 判断 → 再迭代或 finalize)。\n"
    "  • 否则手动调用 scanpy 的 QC 基本指标:\n"
    "      sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], inplace=True)\n"
    "      sc.pp.filter_cells(adata, min_genes=...)\n"
    "      sc.pp.filter_genes(adata, min_cells=...)\n"
    "    然后把 mask / summary / plots 放到对应 attempts/ 目录。"
)


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _get_bash_command(payload: dict) -> Optional[str]:
    tool_name = (
        payload.get("tool_name")
        or (payload.get("tool_use") or {}).get("name")
        or (payload.get("toolUse") or {}).get("name")
    )
    if tool_name != "Bash":
        return None
    tool_input = (
        payload.get("tool_input")
        or (payload.get("tool_use") or {}).get("input")
        or (payload.get("toolUse") or {}).get("input")
        or {}
    )
    cmd = tool_input.get("command")
    return cmd if isinstance(cmd, str) else None


def _exit_code(payload: dict) -> int:
    res = (
        payload.get("tool_response")
        or payload.get("tool_result")
        or payload.get("toolResult")
        or {}
    )
    if not isinstance(res, dict):
        return 0
    for key in ("exit_code", "exitCode", "returncode", "returnCode"):
        code = res.get(key)
        if code is not None:
            try:
                return int(code)
            except (TypeError, ValueError):
                return 1
    if res.get("is_error") or res.get("isError"):
        return 1
    return 0


def main() -> int:
    payload = _read_payload()
    cmd = _get_bash_command(payload)
    if not cmd:
        return 0
    if _exit_code(payload) != 0:
        return 0

    # Split by shell operators so chained commands like
    # ``pip install stancounts && python -c "from stancounts ..."`` still
    # trigger on the run segment while the install segment is skipped.
    matched = False
    for seg in re.split(r"(?:&&|\|\||;|\|)", cmd):
        if _PKG_MGR_CMD.search(seg):
            continue
        if _STANCOUNTS_CMD.search(seg):
            matched = True
            break
    if not matched:
        return 0

    out = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": _HANDOFF,
        }
    }
    try:
        print(json.dumps(out))
    except Exception:
        pass
    try:
        print(_HANDOFF, file=sys.stderr)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
