from dataclasses import dataclass
from pathlib import Path

@dataclass
class InfraSummary:
    traffic_density: str
    pedestrians: int
    signal_state: str
    hazards: list[str]
    evidence: str

def summarize_infra_image(infra_image_path: Path) -> InfraSummary:
    # TODO: 後で GPT/VLM 呼び出しに置き換える
    return InfraSummary(
        traffic_density="medium",
        pedestrians=0,
        signal_state="unknown",
        hazards=["potential occlusion at intersection"],
        evidence=f"dummy summary from {infra_image_path.name}",
    )
