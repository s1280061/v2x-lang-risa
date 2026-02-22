from pathlib import Path
from .infra_summary import InfraSummary

def generate_ego_advice(ego_image_path: Path, infra_summary: InfraSummary) -> str:
    # TODO: 後で ego VLM に置き換える
    return (
        f"[Ego Advice]\n"
        f"- Traffic: {infra_summary.traffic_density}\n"
        f"- Signal: {infra_summary.signal_state}\n"
        f"- Pedestrians: {infra_summary.pedestrians}\n"
        f"- Hazards: {', '.join(infra_summary.hazards)}\n"
        f"- Action: Approach the intersection cautiously and be ready to brake.\n"
        f"(ego={ego_image_path.name})"
    )
