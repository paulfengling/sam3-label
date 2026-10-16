#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "未找到项目运行环境 .venv。请按 RUN_SAM3.md 的安装步骤创建环境。"
    read -r -p "按回车退出…"
    exit 1
fi

exec .venv/bin/python main.py "$@"
