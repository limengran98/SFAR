#!/bin/bash

# 定义 SSH 仓库地址
REPO_SSH="git@github.com:limengran98/SFAR.git"
BRANCH_NAME="main"

echo "========================================"
echo "开始切换到 SSH 并强制同步本地代码"
echo "========================================"

# 1. 清理之前的错误状态
# 如果处于合并冲突状态，先中止合并
if [ -f .git/MERGE_HEAD ]; then
    echo "[操作] 检测到正在进行的合并冲突，正在中止..."
    git merge --abort
    echo "[成功] 已中止失败的合并。"
fi

# 2. 切换远程仓库地址为 SSH
echo "[操作] 设置远程仓库为 SSH 模式: $REPO_SSH"
# 无论是否存在 origin，先尝试删除再添加，或直接设置 URL
if git remote | grep -q "^origin$"; then
    git remote set-url origin "$REPO_SSH"
else
    git remote add origin "$REPO_SSH"
fi

# 3. 提交本地更改
echo "[操作] 添加本地文件..."
git add .

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "[操作] 提交本地更改..."
    git commit -m "Refactor: Modularize project structure (Force Sync)"
else
    echo "[提示] 本地没有新的更改需要提交。"
fi

# 4. 强制推送 (解决 non-fast-forward 问题)
echo "========================================"
echo "正在强制推送到 GitHub (SSH)..."
echo "这将使用本地版本覆盖远程版本。"
echo "========================================"

if git push -u origin $BRANCH_NAME --force; then
    echo "========================================"
    echo "[成功] 代码已通过 SSH 强制上传成功！"
    echo "========================================"
else
    echo "========================================"
    echo "[失败] 推送失败。请检查以下两点："
    echo "1. 你的 GitHub 账户是否已配置 SSH Key (id_rsa.pub)。"
    echo "2. 运行 'ssh -T git@github.com' 测试连接是否通畅。"
    echo "========================================"
    exit 1
fi