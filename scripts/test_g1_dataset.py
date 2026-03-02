#!/usr/bin/env python3
"""G1 データセット読み込みテスト"""
import sys
sys.path.insert(0, "/Users/miyajimatsuyoshifutoshi/dev/Orboh/Isaac-GR00T")

DATASET_PATH = (
    "examples/GR00T-WholeBodyControl/"
    "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/"
    "unitree_g1.LMPnPAppleToPlateDC"
)

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

modality_config = MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1.value]
loader = LeRobotEpisodeLoader(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="cv2",
)
print(f"エピソード数: {len(loader)}")
sample = loader[0]
print("サンプルキー:")
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} ({v.dtype})")
    else:
        print(f"  {k}: {type(v).__name__}")
print("OK: データ読み込み成功")
