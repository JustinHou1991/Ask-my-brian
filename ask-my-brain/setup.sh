#!/bin/bash
# Ask My Brain - 环境初始化脚本
set -e

echo "=============================="
echo "  Ask My Brain - Setup"
echo "=============================="

# 1. 安装依赖
echo "[1/3] 安装 Python 依赖..."
pip install -r requirements.txt

# 2. 预下载 Embedding 模型
echo "[2/3] 预下载 Embedding 模型（约90MB）..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Embedding 模型就绪')"

# 3. 初始化目录
echo "[3/3] 初始化目录结构..."
mkdir -p data/documents chroma_db logs

# 4. 检查 .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  请编辑 .env 文件，填入你的 MIMO_API_KEY"
    echo "  获取密钥: https://platform.xiaomimimo.com → 控制台 → API Key"
fi

echo ""
echo "=============================="
echo "  Setup 完成!"
echo "  启动: make run 或 streamlit run app.py"
echo "=============================="
