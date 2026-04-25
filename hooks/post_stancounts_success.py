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
    "家族流水线中 stancounts 的下游是 **stanqc**(QC stage)。stanqc 是\n"
    "tissue / species / platform 不可知的两层 QC(Tier-1 sample health +\n"
    "Tier-2 per-group MAD),不自动删 cell,只在 obs 加 `out_*` flag 列。\n"
    "\n"
    "建议的下一步:\n"
    "  • 如果当前在 eca-curation pipeline session 里:运行\n"
    "      /eca-run <dataset>\n"
    "    pipeline 会调用 stanqc 推进到 03_qc 阶段(默认 iterative,直到\n"
    "    flagged 比例收敛)。\n"
    "  • 否则直接调 stanqc:\n"
    "      import stanqc\n"
    "      result = stanqc.run_qc(\n"
    "          adata, sample_key=...,\n"
    "          cell_type_key=...,   # 可选,有标注则 per-celltype MAD\n"
    "          species=None,        # None = 自动检测\n"
    "          platform=None,       # None = 自动检测\n"
    "          outdir='qc_out/',\n"
    "      )\n"
    "      # adata.obs 多了 out_* 列;最终是否 drop 由用户决定:\n"
    "      # clean = adata[~adata.obs['out_any']].copy()"
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
