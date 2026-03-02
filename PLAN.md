# GR00T N1.6 Fine-tuning on G1 + AmazingHand: 実装計画

**作成日**: 2026-02-28
**対象モデル**: nvidia/GR00T-N1.6-3B
**対象ロボット**: Unitree G1 body (29 DOF) + AmazingHand 左手 (8 DOF) + AmazingHand 右手 (8 DOF) = **45 DOF**
**Embodiment tag**: `NEW_EMBODIMENT` (projector W[10])

---

## 進捗状況

| Phase | 内容 | 状態 | メモ |
|-------|------|------|------|
| Phase 1 | LeRobot v2 変換 (45 DOF + 動画対応) | ⏳ 待機中 | TWIST2 側の新フォーマットデータ待ち |
| Phase 2 | HuggingFace Hub アップロード | ⏳ 待機中 | Phase 1 完了後 |
| Phase 3 | AWS Fine-tuning | 🔄 進行中 | G1 公開データで動作確認中（scripts/aws/ 準備済み ✅） |
| Phase 4 | Isaac Lab シミュレーション評価 | ⏳ 待機中 | Phase 3 完了後 |
| Phase 5 | 実機 G1 デプロイ | ⏳ 未着手 | |

**凡例**: ✅ 完了 / 🔄 進行中 / ⏳ 待機中 / ❌ ブロック中 / 未着手

### ブロッカー
- ~~**AWS GPU クォータ**: `G and VT instances` を申請済み。承認待ち（2026-02-28 申請）~~ → **承認済み ✅ (2026-03-01)**
- **TWIST2 新フォーマット**: 45 DOF + カメラ対応の JSON が出力できるようになったら Phase 1 着手

---

## このリポジトリのスコープ

```
【対象外: TWIST2 リポジトリ側】
  TWIST2 で G1 + AmazingHand + カメラを記録
  → episodes-twist2/*.json + カメラ映像 を出力

【このリポジトリの担当範囲 ↓】
  ┌─ Phase 1: LeRobot v2 変換
  │    scripts/convert_twist2_to_lerobot.py (更新)
  │    → datasets/twist2_lerobot_v2/
  ├─ Phase 2: HuggingFace Hub アップロード
  │    → hf://datasets/<org>/twist2-g1-amazinghand
  ├─ Phase 3: AWS Fine-tuning
  │    gr00t/experiment/launch_finetune.py
  │    → checkpoints/ (S3 バックアップ)
  ├─ Phase 4: Isaac Lab シミュレーション評価
  │    gr00t/eval/run_gr00t_server.py + rollout_policy.py
  └─ Phase 5: 実機 G1 デプロイ
       gr00t/eval/run_gr00t_server.py on Jetson AGX Thor
```

---

## 前提: TWIST2 が出力する入力フォーマット

TWIST2 側で 45 DOF + カメラに対応済みである前提。
このリポジトリが受け取る期待フォーマット:

**JSON (フレームごと)**:
```json
{
  "schema_version": "1.1",
  "timestamp": 1234567890.123456,
  "body_dof_pos":              [float x 29],
  "target_body_dof_pos":       [float x 29],
  "left_hand_dof_pos":         [float x 8],
  "target_left_hand_dof_pos":  [float x 8],
  "right_hand_dof_pos":        [float x 8],
  "target_right_hand_dof_pos": [float x 8],
  "tau":         [float x 29],
  "voltage":     [float x 29],
  "temperature": [[float, float] x 29]
}
```

**observation.state / action の全体レイアウト (45 DOF)**:
```
indices  0-28: body        (G1 body 29 DOF)
indices 29-36: left_hand   (AmazingHand 左手 8 DOF)
indices 37-44: right_hand  (AmazingHand 右手 8 DOF)
```

**ディレクトリ構成（カメラあり）**:
```
episodes-twist2/YYYY-MM-DD/
├── twist2_real_recordings_YYYYMMDD_HHMMSS.json
├── camera_YYYYMMDD_HHMMSS.mp4
└── camera_timestamps_YYYYMMDD_HHMMSS.json  # {frame_idx: timestamp}
```

> **注意**: 現在の `episodes-twist2/` にある JSON は 29 DOF のみ・カメラなしの旧形式。
> 新形式での収録が揃ってから変換を実行する。

---

## Phase 1: LeRobot v2 変換

### 1.1 変更が必要なファイル

#### [scripts/convert_twist2_to_lerobot.py](scripts/convert_twist2_to_lerobot.py)

**現在の制限**:
- `DOF_COUNT = 29`（body のみ）
- 動画未対応（`total_videos: 0`）
- `build_modality_json()` に hand / video エントリがない

**必要な変更の要点**:

```python
# 定数の更新
BODY_DOF = 29
HAND_DOF = 8           # 片手あたり
TOTAL_DOF = BODY_DOF + HAND_DOF * 2  # 45

# observation.state / action を 45 次元に
"observation.state": np.concatenate([
    np.array(frame["body_dof_pos"],       dtype=np.float32),  # [29]
    np.array(frame["left_hand_dof_pos"],  dtype=np.float32),  # [8]
    np.array(frame["right_hand_dof_pos"], dtype=np.float32),  # [8]
]),
"action": np.concatenate([
    np.array(frame["target_body_dof_pos"],       dtype=np.float32),
    np.array(frame["target_left_hand_dof_pos"],  dtype=np.float32),
    np.array(frame["target_right_hand_dof_pos"], dtype=np.float32),
]),

# build_modality_json() の更新
{
    "state": {
        "body":       {"start": 0,  "end": 29},
        "left_hand":  {"start": 29, "end": 37},
        "right_hand": {"start": 37, "end": 45}
    },
    "action": {
        "body":       {"start": 0,  "end": 29},
        "left_hand":  {"start": 29, "end": 37},
        "right_hand": {"start": 37, "end": 45}
    },
    "video": {
        "ego_view": {"original_key": "observation.images.ego_view"}
    },
    "annotation": {"human.action.task_description": {}}
}
```

**動画変換処理の追加**:
```python
import subprocess

def convert_video_to_mp4(src_path, dst_path, fps):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src_path),
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst_path)
    ], check=True)

# 出力先: videos/chunk-{chunk:03d}/observation.images.ego_view/episode_{ep:06d}.mp4
```

#### [configs/data/twist2_modality_config.py](configs/data/twist2_modality_config.py)

```python
BODY_DOF = 29
HAND_DOF = 8
ACTION_HORIZON = 16

twist2_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["body", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["body", "left_hand", "right_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="body",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_hand",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_hand",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(twist2_config, EmbodimentTag.NEW_EMBODIMENT)
```

**設計理由**:
- body (G1 29 DOF) は `RELATIVE` — スムーズな全身動作の学習に適する
- left_hand / right_hand (各 8 DOF) は `ABSOLUTE` — グリッパー的なポジション制御（UNITREE_G1 の hand 設定と整合）

### 1.2 変換コマンド

```bash
uv run python scripts/convert_twist2_to_lerobot.py \
  --input-dir episodes-twist2 \
  --output-dir datasets/twist2_lerobot_v2 \
  --task-description "G1 with AmazingHand manipulation" \
  --fps 50 \
  --episode-length 200
```

### 1.3 変換後の検証

```bash
uv run python - <<'EOF'
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
import importlib, sys
sys.path.append("configs/data")
importlib.import_module("twist2_modality_config")
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
modality = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
ds = LeRobotEpisodeLoader("datasets/twist2_lerobot_v2", modality)
print(f"エピソード数: {len(ds)}")
EOF
```

---

## Phase 2: HuggingFace Hub アップロード

```bash
# 認証
huggingface-cli login

# リポジトリ作成（初回のみ）
huggingface-cli repo create twist2-g1-amazinghand --type dataset --private

# アップロード
huggingface-cli upload <YOUR_HF_USERNAME>/twist2-g1-amazinghand \
  datasets/twist2_lerobot_v2 \
  --repo-type dataset \
  --commit-message "Add 45DOF + camera dataset (G1 + AmazingHand)"
```

**注意**: `.mp4` は HuggingFace Hub が自動で LFS 管理する。

---

## Phase 3: AWS Fine-tuning

### 3.1 推奨インスタンス

| インスタンス | GPU | VRAM | Spot 概算 | 推奨 |
|-------------|-----|------|-----------|------|
| **g5.12xlarge** | 4x A10G 24GB | 96GB | ~$1.7/時 | コスト効率重視 |
| **p4d.24xlarge** | 8x A100 80GB | 640GB | ~$10/時 | 速度重視 |

Spot Instance を利用してコストを削減。チェックポイントを S3 に頻繁に保存して中断に備えること。

### 3.2 EC2 セットアップ

```bash
# Deep Learning AMI (Ubuntu 22.04) 起動後
git clone https://github.com/NVlabs/Isaac-GR00T.git
cd Isaac-GR00T
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv sync --python 3.10 && uv pip install -e .

# HuggingFace 認証とデータセット取得
huggingface-cli login
huggingface-cli download <YOUR_HF_USERNAME>/twist2-g1-amazinghand \
  --repo-type dataset \
  --local-dir /datasets/twist2_lerobot_v2
```

### 3.3 Fine-tuning 実行

**シングル GPU (g5.12xlarge 等):**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /datasets/twist2_lerobot_v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path configs/data/twist2_modality_config.py \
  --num-gpus 1 \
  --output-dir /checkpoints/twist2_finetune \
  --save-total-limit 5 \
  --save-steps 1000 \
  --max-steps 10000 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --global-batch-size 64 \
  --use-wandb
```

**マルチ GPU (p4d.24xlarge: 8x A100):**
```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /datasets/twist2_lerobot_v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path configs/data/twist2_modality_config.py \
  --num-gpus 8 \
  --output-dir /checkpoints/twist2_finetune \
  --global-batch-size 1024 \
  --use-wandb
```

**デフォルトで tune される対象**:
- `tune_projector=True` — projector W[10] を学習（必須）
- `tune_diffusion_model=True` — DiT action head を学習（必須）
- `tune_llm=False` — LLM backbone は凍結（VRAM 節約）
- `tune_visual=False` — visual encoder は凍結（VRAM 節約）

### 3.4 チェックポイントの S3 保存

```bash
aws s3 sync /checkpoints/twist2_finetune \
  s3://gr00t-finetune-checkpoints/twist2_finetune \
  --exclude "*.tmp"
```

### 3.5 Open-loop 事前評価

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /datasets/twist2_lerobot_v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /checkpoints/twist2_finetune/checkpoint-10000 \
  --traj-ids 0 1 2 \
  --action-horizon 16
```

---

## Phase 4: Isaac Lab シミュレーション評価

### 4.1 既存環境のセットアップ

```bash
apt-get update && apt-get install -y libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

**オープンクエスチョン [Q3]**: Isaac Lab に G1+AmazingHand 環境はあるか？
- ある場合: そのまま利用
- ない場合の暫定策: hand DOF を固定値（全開）にして G1 body の 29 DOF のみで評価

### 4.2 評価手順

**Terminal 1 - ポリシーサーバー:**
```bash
aws s3 sync s3://gr00t-finetune-checkpoints/twist2_finetune/checkpoint-10000 \
  /tmp/twist2_checkpoint

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /tmp/twist2_checkpoint \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 --host 0.0.0.0 --port 5555
```

**Terminal 2 - シミュレーションクライアント:**
```bash
uv run python gr00t/eval/rollout_policy.py \
  --n_episodes 10 \
  --max_episode_steps 1440 \
  --n_action_steps 16 \
  --n_envs 1
```

---

## Phase 5: 実機 G1 デプロイ

### 5.1 チェックポイントの転送

```bash
aws s3 sync s3://gr00t-finetune-checkpoints/twist2_finetune/checkpoint-10000 \
  ~/models/twist2_checkpoint
```

### 5.2 ポリシーサーバー起動（Jetson AGX Thor 等）

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path ~/models/twist2_checkpoint \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 --host 0.0.0.0 --port 5555
```

### 5.3 G1 制御クライアント（TWIST2 側に実装）

```python
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="<JETSON_IP>", port=5555)
client.ping()

obs = {
    "video": {"ego_view": camera_frame[None, None, ...]},
    "state": {
        "body":       body_state[None, None, ...],        # (1,1,29)
        "left_hand":  left_hand_state[None, None, ...],   # (1,1,8)
        "right_hand": right_hand_state[None, None, ...],  # (1,1,8)
    },
    "language": {
        "annotation.human.action.task_description": [["task description here"]]
    }
}

action, _ = client.get_action(obs)
body_action       = action["body"][0]        # (action_horizon, 29)
left_hand_action  = action["left_hand"][0]   # (action_horizon, 8)
right_hand_action = action["right_hand"][0]  # (action_horizon, 8)
```

### 5.4 安全チェックリスト

- [ ] 各 DOF のソフトウェアリミット（クリッピング）設定済み
- [ ] 緊急停止ボタンの動作確認
- [ ] 最初の実行はスピードスケール 0.1〜0.2 で行う
- [ ] シミュレーションで十分な成功率を確認してからデプロイ

---

## 重要ファイル一覧

| ファイル | 役割 | 現状 |
|---------|------|------|
| [scripts/convert_twist2_to_lerobot.py](scripts/convert_twist2_to_lerobot.py) | LeRobot v2 変換 | 29 DOF, 動画なし → **要更新** |
| [configs/data/twist2_modality_config.py](configs/data/twist2_modality_config.py) | Modality 設定 | 29 DOF, 動画なし → **要更新** |
| [gr00t/experiment/launch_finetune.py](gr00t/experiment/launch_finetune.py) | Fine-tuning エントリ | 変更不要 |
| [gr00t/eval/run_gr00t_server.py](gr00t/eval/run_gr00t_server.py) | ZMQ ポリシーサーバー | 変更不要 |
| [gr00t/eval/open_loop_eval.py](gr00t/eval/open_loop_eval.py) | Open-loop 評価 | 変更不要 |
| [gr00t/eval/rollout_policy.py](gr00t/eval/rollout_policy.py) | Closed-loop 評価 | 変更不要 |

## オープンクエスチョン

| ID | 質問 | ブロッカー? |
|----|------|-----------|
| Q1 | TWIST2 が出力する新フォーマット (45 DOF + camera) の確定 | Phase 1 の前提 |
| Q2 | カメラ映像の同期方式（別 MP4 + タイムスタンプ JSON が推奨） | Phase 1 の前提 |
| Q3 | Isaac Lab に G1+AmazingHand 環境はあるか？ | Phase 4 の実現可否 |
