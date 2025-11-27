#!/bin/bash

# 定义仓库地址
REPO_URL="https://github.com/limengran98/SFAR.git"
BRANCH_NAME="main"

echo "========================================"
echo "开始准备上传代码到: $REPO_URL"
echo "========================================"

# 1. 检查并创建 .gitignore (防止上传垃圾文件和大文件)
if [ ! -f .gitignore ]; then
    echo "[信息] 未检测到 .gitignore，正在创建..."
    cat <<EOT >> .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class

# Data & Models (防止大文件导致上传失败)
*.pt
*.pth
*.pkl
embeddings/
LLMs/
data/

# OS
.DS_Store
.idea/
.vscode/
EOT
    echo "[成功] 已创建 .gitignore"
else
    echo "[信息] .gitignore 已存在，跳过创建。"
fi

# 2. 初始化 Git
if [ ! -d .git ]; then
    echo "[操作] 初始化 Git 仓库..."
    git init
else
    echo "[信息] Git 仓库已初始化。"
fi

# 3. 规范化分支名称
git branch -M $BRANCH_NAME

# 4. 配置远程仓库
# 先尝试移除旧的 origin (如果存在)，确保指向正确的 URL
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
echo "[成功] 远程仓库已设置为 $REPO_URL"

# 5. 添加文件并提交
echo "[操作] 添加所有文件到暂存区..."
git add .

# 检查是否有文件需要提交
if [ -n "$(git status --porcelain)" ]; then
    echo "[操作] 提交文件..."
    # 使用时间戳作为 commit message，防止重复提交报错
    git commit -m "Auto-upload: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "[提示] 没有文件发生变化，无需提交。"
fi

# 6. 推送代码
echo "========================================"
echo "正在推送到 GitHub，请稍候..."
echo "如果这是第一次连接，可能需要输入 GitHub 用户名和 Token (或密码)"
echo "========================================"

# 尝试推送 (如果远程有冲突，尝试拉取合并后再推送)
if git push -u origin $BRANCH_NAME; then
    echo "========================================"
    echo "[成功] 代码已上传!"
    echo "========================================"
else
    echo "========================================"
    echo "[警告] 直接推送失败，尝试拉取远程更新并合并..."
    echo "========================================"
    git pull origin $BRANCH_NAME --allow-unrelated-histories --no-rebase
    
    echo "[操作] 再次尝试推送..."
    if git push -u origin $BRANCH_NAME; then
        echo "========================================"
        echo "[成功] 代码已合并并上传!"
        echo "========================================"
    else
        echo "[错误] 上传失败。请检查网络或是否需要 'git push --force' (慎用)。"
        exit 1
    fi
fi