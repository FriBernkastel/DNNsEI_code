"""Final integrity check: dollar-sign parity, escapes, and CJK-in-math."""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MD = Path(r"D:\我的文档\OneDrive\Desktop\连续EI奇异值凹陷与噪声下限修正分析记录.md")
lines = MD.read_text(encoding="utf-8").split("\n")

bad_parity = []
in_fence = False
for n, line in enumerate(lines, 1):
    if line.strip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    # ignore $ inside inline code spans
    stripped = re.sub(r"`[^`]*`", "", line)
    cnt = stripped.count("$")
    if cnt % 2 != 0:
        bad_parity.append((n, cnt, line[:60]))
print("lines with odd $ count:", bad_parity if bad_parity else "none")

text = "\n".join(lines)
print(r"backslash-underscore occurrences:", text.count(r"\_"))
for m in re.finditer(r"\\text\{([^}]*)\}", text):
    if re.search(r"[\u4e00-\u9fff]", m.group(1)):
        print("CJK in \\text{}:", m.group(1))
print("inline $...$ on one line:", len(re.findall(r"(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)", text)))
print("display blocks:", len([i for i, l in enumerate(lines) if l.strip() == "$$"]) // 2)
print("file bytes:", len(text.encode("utf-8")), "| lines:", len(lines))
