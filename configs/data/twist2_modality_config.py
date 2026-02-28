"""
Modality configuration for Twist2 robot (no vision, 29 DOF).

This file is loaded via --modality-config-path in launch_finetune.py.
Vision modalities will be added here when camera data becomes available.

DOF layout (29 total):
    body: indices 0-28  (all 29 joint positions)

Usage:
    uv run python gr00t/experiment/launch_finetune.py \
        --base-model-path nvidia/GR00T-N1.6-3B \
        --dataset-path datasets/twist2_lerobot \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path configs/data/twist2_modality_config.py \
        --num-gpus 1 --output-dir <OUTPUT_PATH>
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

DOF_COUNT = 29
ACTION_HORIZON = 16  # predict 16 future steps at once

twist2_config = {
    # No video modality until vision is integrated
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="observation.state",
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(twist2_config, EmbodimentTag.NEW_EMBODIMENT)
