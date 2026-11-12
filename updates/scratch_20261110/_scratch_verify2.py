"""Precise check: inspect the actual bodies between delimiter lines."""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MD = Path(r"D:\我的文档\OneDrive\Desktop\连续EI奇异值凹陷与噪声下限修正分析记录.md")
lines = MD.read_text(encoding="utf-8").split("\n")
delims = [i for i, l in enumerate(lines) if l.strip() == "$$"]

ok = True
for k in range(0, len(delims), 2):
    start, end = delims[k], delims[k + 1]
    body = "\n".join(lines[start + 1 : end]).strip()
    cjk = re.findall(r"[\u4e00-\u9fff]+", body)
    if cjk:
        ok = False
        print(f"BLOCK {k//2+1} (line {start+1}-{end+1}) has CJK: {cjk[:5]}")
    else:
        print(f"BLOCK {k//2+1} (line {start+1}-{end+1}) OK  | {body[:70]}")
print("\nall display bodies ASCII-math only:", ok)
