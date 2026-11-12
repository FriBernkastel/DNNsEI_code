# 版本管理（Git）使用说明

> 目标：**每改一次都能存一版；任何时候都能回到过去某一版；重要文件不会因为一次误删而消失。**

## 0. 现在有什么

- 仓库已经建在 `DNNsEI_code_20261029/`（隐藏目录 `.git`），**171 个文件已入库**，首次提交：
  `87310e9 baseline: 2026-11-12 现状快照`
- 工作区当前是干净的（`git status` 无输出）。
- 之前的原始 zip 已不在工作区，所以**历史从今天开始**；从今天起每次改动都可回滚。
- 提交身份目前是占位值，建议改成你自己的（只影响提交记录里的作者名）：

```
git config user.name  "你的名字"
git config user.email "you@example.com"
```

---

## 1. 每次改完存一版（最重要的一条）

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\save.ps1 "说明这次改了什么"
```

`save.ps1` 会自动：`git add -A` → 列出改动文件 → 提交（自动带时间戳）→ 显示最近 3 条历史。

**在 VSCode 里**：左侧「源代码管理」图标 → 输入消息 → 点 ✓ 提交，效果相同。
建议节奏：*每改完一个功能 / 每跑完一次实验 / 每次要尝试有风险的改动之前*，各存一版。

---

## 2. 看历史：我改过什么

```
git log --oneline --graph          # 提交列表（一行一条）
git show <commit> --stat           # 某次提交改了哪些文件
git diff <commit1> <commit2>       # 两次提交之间的全部差异
git diff <commit> -- eicore/continuous.py   # 只看某个文件
git blame eicore/continuous.py     # 每一行是哪个提交改的
```

提交号（如 `87310e9`）就是版本号，可以只写前 7 位。

---

## 3. 回滚的四种姿势（按危险程度排序）

| 场景 | 命令 | 说明 |
|---|---|---|
| 只想**看**旧版本，不动现在 | `git switch -c look-old 87310e9` | 新建分支看，看完 `git switch main` 回来 |
| 恢复**单个文件**到某版 | `git checkout 87310e9 -- eicore/continuous.py` | 只覆盖这一个文件，其他不动 |
| 撤销**未提交**的改动 | `git restore <文件>` | 回到上次提交的状态 |
| 整个项目回到某版（**保留**历史） | `git switch -c back-1112 87310e9` | 旧版本变成新分支，main 不受影响 |
| 整个项目回到某版（**丢弃**之后改动） | `git reset --hard 87310e9` | ⚠️ 危险，之后仍可用 reflog 救回 |

**推荐**：回滚前先 `git switch -c try-xxx` 开分支，确认没问题再合并回 `main`，这样永远不会"删掉重要文件"。

---

## 4. 里程碑：给重要版本打标签

```
git tag -a v1.0 -m "第一次完整复现（图 2/4 对齐论文）"
git tag                     # 列出所有标签
git show v1.0               # 看标签对应的提交
git switch -c from-v1 v1.0  # 从标签开分支
```

标签比提交号好记，适合"论文投稿版""答辩版"这类节点。

---

## 5. 手滑删错了怎么救

```
git reflog                  # 最近 30 天所有 HEAD 移动记录（包括已被 reset 掉的提交）
git reset --hard HEAD@{2}   # 回到 reflog 里的第 2 条
```

`reflog` 是本地"后悔药"，**只有在 `git gc` 清理且超过 90 天后**才会消失。

---

## 6. 远程备份（GitHub / Gitee）

本地 git 已经能回滚；远程只解决"磁盘坏了/换电脑"。步骤：

**GitHub**
1. 网页 New repository，**不要**勾选初始化 README（否则要先 pull）。
2. 在项目里执行：
```
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```
3. 推送需要凭据（Personal Access Token）。不要把 token 写进脚本或截图；推荐用 Git 凭据管理器或 `gh auth login`。

**国内网络备选（Gitee）**
```
git remote add origin https://gitee.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

> 如果 `git push` 卡住或超时，多半是网络问题：可换 Gitee，或先做下面的离线备份。

**离线备份（不依赖任何服务，最稳）**
```
git bundle create D:\backup\DNNsEI_20261112.bundle --all
```
恢复：
```
git clone D:\backup\DNNsEI_20261112.bundle DNNsEI_restored
```
建议每周做一次 bundle，放到 U 盘/网盘。

---

## 7. 关于 `results/` 与体积

现在 `results/`（约 10 MB 的图与 CSV）**已纳入版本控制**，因为它们是实验记录。
如果以后仓库太大（`git count-objects -vH` 看大小）：

```
git rm -r --cached results
echo results/ >> .gitignore
git commit -m "stop tracking generated results"
```

之后新跑的结果不再入库，历史版本仍可查。

---

## 8. 一页速查

```
# 存一版
powershell -NoProfile -ExecutionPolicy Bypass -File tools\save.ps1 "改了什么"

# 看历史 / 看差异
git log --oneline --graph
git show <hash> --stat

# 回滚单个文件
git checkout <hash> -- <path>

# 回滚整个项目（安全做法）
git switch -c back-<date> <hash>

# 打标签
git tag -a v1 -m "里程碑"

# 误删救命
git reflog

# 远程 / 离线备份
git push -u origin main
git bundle create D:\backup\DNNsEI_<date>.bundle --all
```
