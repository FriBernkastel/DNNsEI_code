"""GitHub-flavoured inline-math compliance scan.

GitHub (and several other parsers) only recognise ``$...$`` when:
  * the opening ``$`` is not followed by whitespace,
  * the closing ``$`` is not preceded by whitespace,
  * the closing ``$`` is not followed by a digit.
Anything else is left as literal text.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MD = Path(r"D:\我的文档\OneDrive\Desktop\连续EI奇异值凹陷与噪声下限修正分析记录.md")
lines = MD.read_text(encoding="utf-8").split("\n")

VALID = re.compile(r"(?<![\$\\])\$(?![\$\s])([^\$\n]+?)(?<![\s\\])\$(?![\$\d])")

in_fence = False
suspicious = []
for n, line in enumerate(lines, 1):
    if line.strip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    stripped = re.sub(r"`[^`]*`", "", line)          # inline code is not math
    stripped = stripped.replace("$$", "")             # display delimiters
    leftover = VALID.sub("", stripped)
    if "$" in leftover:
        suspicious.append((n, leftover.strip()[:80]))

print("lines with non-compliant inline math:", len(suspicious))
for n, frag in suspicious[:20]:
    print(f"  L{n}: {frag}")
