#!/usr/bin/env bash
# AWS EC2 セットアップスクリプト
# 対象: Deep Learning AMI (Ubuntu 22.04) on g5.12xlarge / p4d.24xlarge
# 使い方: bash scripts/aws/setup_ec2.sh
set -euo pipefail

echo "=== [1/5] uv のインストール ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== [2/5] リポジトリのクローン ==="
cd ~
git clone https://github.com/NVlabs/Isaac-GR00T.git
cd Isaac-GR00T

echo "=== [3/5] 依存関係インストール ==="
uv pip install -e .
uv pip install -e .[dev]

echo "=== [4/5] HuggingFace 認証 ==="
echo ">>> HuggingFace トークンを入力してください（読み取り権限があれば OK）"
uv run huggingface-cli login

echo "=== [5/5] G1 データセットのダウンロード ==="
uv run python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    repo_type="dataset",
    allow_patterns=["unitree_g1.LMPnPAppleToPlateDC/**"],
    local_dir="examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    ignore_patterns=["*.git*"],
)
print("Dataset download complete.")
PYEOF

echo ""
echo "=== セットアップ完了 ==="
echo "次のコマンドで fine-tuning を開始:"
echo "  bash scripts/aws/finetune_g1_aws.sh"
