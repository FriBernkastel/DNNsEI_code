"""Verify math formatting of the Desktop report (fixed checker)."""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MD = Path(r"D:\我的文档\OneDrive\Desktop\连续EI奇异值凹陷与噪声下限修正分析记录.md")
text = MD.read_text(encoding="utf-8")
lines = text.split("\n")

delims = [i for i, l in enumerate(lines) if l.strip() == "$$"]
print("delimiter lines:", len(delims), "-> blocks:", len(delims) // 2, "| balanced:", len(delims) % 2 == 0)

problems = []
for k, i in enumerate(delims):
    if k % 2 == 0:  # opening
        prev = lines[i - 1] if i > 0 else ""
        if prev.strip():
            problems.append((i + 1, "opening delimiter needs a blank line before", prev[:50]))
    else:  # closing
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt.strip():
            problems.append((i + 1, "closing delimiter needs a blank line after", nxt[:50]))
print("blank-line problems:", problems if problems else "none")

# strip fenced code blocks
no_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

# CJK inside display math
body_cjk = []
for m in re.finditer(r"\$\$(.*?)\$\$", no_code, flags=re.DOTALL):
    body = m.group(1)
    if re.search(r"[\u4e00-\u9fff]", body):
        body_cjk.append(body.strip()[:60])
print("display-math bodies containing CJK:", body_cjk if body_cjk else "none")

# CJK inside inline math (single line only)
inline_cjk = []
for line in no_code.split("\n"):
    for m in re.finditer(r"(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)", line):
        if re.search(r"[\u4e00-\u9fff]", m.group(1)):
            inline_cjk.append(m.group(1)[:40])
print("inline-math fragments containing CJK:", inline_cjk if inline_cjk else "none")

for macro in [r"\underbrace", r"\tfrac", r"\Big", r"\big(", r"\left(", r"\right)"]:
    print(f"  {macro!r}: {text.count(macro)}")
print("single-line $$...$$:", len(re.findall(r"^\$\$.+\$\$$", text, flags=re.MULTILINE)))
print("inline math count:", len(re.findall(r"(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)", no_code)))
