"""G1 WholeBodyControl データセットを HuggingFace からダウンロードするスクリプト"""
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    repo_type="dataset",
    allow_patterns=["unitree_g1.LMPnPAppleToPlateDC/**"],
    local_dir="examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    ignore_patterns=["*.git*"],
)
print("Dataset download complete.")
