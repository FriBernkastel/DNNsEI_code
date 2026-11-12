"""把一次运行的产物镜像到报告专用的固定路径 ``results/latest/<实验名>/``。

背景：实验每次都把结果写进带时间戳的新目录（``iris_continuous_ei_20261104_201943``
之类），而说明文档（``experiment_figures_*.html``）只能写死图片路径。若文档直接
引用时间戳目录，重跑一次就全部 404。因此这里把最新一次运行的 png/csv 复制到固定
别名目录，文档只引用 ``results/latest/<实验名>/`` 即可长期有效。

镜像只复制图片和 CSV，不复制其他中间产物；覆盖前会清掉别名目录里同后缀的旧文件，
避免上一次运行残留的文件冒充本次结果。
"""

import re
import shutil
from pathlib import Path

LATEST_DIRNAME = "latest"
MIRROR_SUFFIXES = (".png", ".csv")

_RUN_DIR_RE = re.compile(r"^(?P<prefix>.+)_(?P<stamp>\d{8}_\d{6})$")


def run_alias(run_dir):
    """``iris_continuous_ei_20261104_201943`` -> ``iris_continuous_ei``。"""
    name = Path(run_dir).name
    match = _RUN_DIR_RE.match(name)
    return match.group("prefix") if match else name


def publish_latest(run_dir, outdir="results"):
    """把 ``run_dir`` 里的图片/CSV 镜像到 ``<outdir>/latest/<别名>/``。"""
    run_dir = Path(run_dir)
    target = Path(outdir) / LATEST_DIRNAME / run_alias(run_dir)
    if target.resolve() == run_dir.resolve():
        return target
    target.mkdir(parents=True, exist_ok=True)

    for stale in target.iterdir():
        if stale.is_file() and stale.suffix in MIRROR_SUFFIXES:
            stale.unlink()

    for item in sorted(run_dir.iterdir()):
        if item.is_file() and item.suffix in MIRROR_SUFFIXES:
            shutil.copy2(item, target / item.name)
    return target


def publish_file(path, outdir="results"):
    """把单个文件（如两口径对比图）镜像到 ``<outdir>/latest/``。"""
    path = Path(path)
    target = Path(outdir) / LATEST_DIRNAME / path.name
    if target.resolve() == path.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def publish_all(results_dir="results"):
    """扫描 ``results_dir``，把每个实验前缀下最新的一次运行镜像到 latest/。"""
    results = Path(results_dir)
    if not results.is_dir():
        return {}

    newest = {}
    for path in sorted(results.iterdir()):
        if not path.is_dir() or path.name == LATEST_DIRNAME:
            continue
        match = _RUN_DIR_RE.match(path.name)
        if not match:
            continue
        prefix = match.group("prefix")
        stamp = match.group("stamp")
        if prefix not in newest or stamp > newest[prefix][0]:
            newest[prefix] = (stamp, path)

    published = {}
    for prefix in sorted(newest):
        published[prefix] = publish_latest(newest[prefix][1], results)
    for png in sorted(results.glob("*.png")):
        published[png.name] = publish_file(png, results)
    return published
