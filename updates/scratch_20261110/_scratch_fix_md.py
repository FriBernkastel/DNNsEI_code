"""Normalize math in the Desktop report so common Markdown renderers typeset it.

Root cause: all 13 display equations were written as single-line ``$$...$$``.
Several renderers (VS Code built-in preview, Typora, Obsidian) only treat
``$$`` as block math when the delimiters stand on their own lines.
"""
from __future__ import annotations

import re
from pathlib import Path

MD = Path(r"D:\我的文档\OneDrive\Desktop\连续EI奇异值凹陷与噪声下限修正分析记录.md")

text = MD.read_text(encoding="utf-8")
original = text

# 1) The only equation that relies on \underbrace with CJK captions: rewrite plainly.
underbrace_eq = (
    r"$$\mathrm{EI} = \frac{1}{n_{\text{in}}}\Big[\underbrace{\text{term1}}_{\text{噪声成本}}"
    r" + \underbrace{H(X)}_{\text{输入熵}} + \underbrace{\mathbb{E}\log\lvert\det J\rvert}_{\text{信号体积}}\Big]$$"
)
underbrace_new = "\n".join(
    [
        "$$",
        r"\mathrm{EI} = \frac{1}{n_{\text{in}}} \left[ \mathrm{term1} + H(X) + \mathbb{E}\log\lvert\det J\rvert \right]",
        "$$",
        "",
        "其中 term1 为噪声成本项，$H(X)$ 为输入熵项，$\\mathbb{E}\\log\\lvert\\det J\\rvert$ 为信号体积项。",
    ]
)
if underbrace_eq in text:
    text = text.replace(underbrace_eq, underbrace_new)

# 2) Every remaining single-line $$...$$ becomes a three-line block.
single_line = re.compile(r"^\$\$(.+?)\$\$\s*$", re.MULTILINE)


def to_block(match: re.Match) -> str:
    body = match.group(1).strip()
    return "$$\n" + body + "\n$$"


text = single_line.sub(to_block, text)

# 3) Replace macros that a few renderers do not support, or that react badly to CJK.
replacements = {
    r"\tfrac": r"\frac",
    r"\Big[": r"\left[",
    r"\Big]": r"\right]",
    r"\big(": r"\left(",
    r"\big)": r"\right)",
    r"\text{阈值}": r"\text{threshold}",
    r"\text{floor\_gain}": r"\text{floor gain}",
    r"\text{tax\_fix}": r"\text{tax fix}",
}
for old, new in replacements.items():
    text = text.replace(old, new)

# 3b) One table header keeps a CJK glyph inside math; move it out of math mode.
text = text.replace("| epoch | term1 | term2 | term1+term2 | EI $=(\\text{和})/5$ |",
                    "| epoch | term1 | term2 | term1+term2 | EI（和 / 5） |")

# 4) Document the math convention right after the notation block.
anchor = '- **关于比喻的说明**'
note = (
    "- **公式渲染约定**：块级公式一律以独占一行的 `$$` 包裹（兼容 VS Code、Typora、"
    "Obsidian、GitHub），行内公式用 `$...$`；不使用 `\\underbrace` 等对中文不友好的构造。\n"
)
if "公式渲染约定" not in text:
    idx = text.find(anchor)
    if idx != -1:
        text = text[:idx] + note + text[idx:]

# 5) Guarantee a blank line before and after every display block. Several parsers
#    only close a ``$$`` block when the following line is blank; otherwise the text
#    after the formula is swallowed and the formula is shown verbatim.
lines = text.split("\n")
out: list[str] = []
i = 0
while i < len(lines):
    if lines[i].strip() == "$$":
        j = i + 1
        while j < len(lines) and lines[j].strip() != "$$":
            j += 1
        if j < len(lines):
            body = lines[i + 1 : j]
            while body and body[-1].strip() == "":
                body.pop()
            if out and out[-1].strip() != "":
                out.append("")
            out.append("$$")
            out.extend(body)
            out.append("$$")
            if j + 1 < len(lines) and lines[j + 1].strip() != "":
                out.append("")
            i = j + 1
            continue
    out.append(lines[i])
    i += 1
text = "\n".join(out)

MD.write_text(text, encoding="utf-8")

# report
remaining = re.findall(r"^\$\$.+\$\$\s*$", text, flags=re.MULTILINE)
print("changed:", text != original)
print("single-line $$ blocks left:", len(remaining))
print("display blocks:", text.count("\n$$\n"))
print("bytes:", len(text.encode("utf-8")))
print("tfrac left:", text.count(r"\tfrac"), "| Big[ left:", text.count(r"\Big["), "| underbrace left:", text.count(r"\underbrace"))
