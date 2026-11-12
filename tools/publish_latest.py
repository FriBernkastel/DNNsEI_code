"""手动刷新报告固定路径：把 results/ 下每个实验最新的运行镜像到 results/latest/。

实验脚本跑完会自己调用 ``eicore.report.publish_latest``；这个入口用于补刷，
例如换了 outdir、或旧结果还没镜像过的时候：

    python -m tools.publish_latest
"""

import argparse

from eicore.report import publish_all


def main():
    parser = argparse.ArgumentParser(description="把 results/ 下最新一次运行镜像到 results/latest/。")
    parser.add_argument("--results", type=str, default="results", help="results 目录（默认 results）")
    args = parser.parse_args()

    published = publish_all(args.results)
    if not published:
        print(f"{args.results} 下没有找到可发布的运行目录")
        return
    for name, path in sorted(published.items()):
        print(f"{name} -> {path}")


if __name__ == "__main__":
    main()
