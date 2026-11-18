# 이 파일이 하는 일: 행사장 약도(길과, 길 위에 떨어져 놓인 구역 A/B/C와 구역별 CCTV)를 읽고, 각 CCTV가 찾은 사람의
#   발 위치가 "세는 바닥" 안에 있는지 가리고, 그 자리를 약도 위의 점으로 옮긴다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-02, C-23, C-24, C-37
# 근거: D-35 (도면 위의 구역과 사람 점), D-38 (도면 안에 CCTV 범위를 놓음), D-42 (구역은 네모),
#   D-44 (AI Hub 인파밀집 영상, 세 거리를 구역으로), D-45 (구역이 이어져 있다고 가정, 영상의 차례),
#   D-46 (약도의 모양: 길로 이어지고 구역은 떨어져 있음),
#   E-37 (기준점은 발 위치), E-47 (세는 바닥의 네 점)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crowdflow.crowd_vision import load_yaml

ROOT = Path(__file__).resolve().parents[2]
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


@dataclass
class Cctv:
    """CCTV 한 대: 화면에서 사람을 세는 바닥(네 점)과, 차례로 잇는 영상들."""

    id: str
    image_size: tuple[int, int]
    floor_quad_px: np.ndarray  # (4, 2): 먼 쪽 왼쪽, 먼 쪽 오른쪽, 가까운 쪽 오른쪽, 가까운 쪽 왼쪽
    clips: list[str]
    video_dir: Path
    detections_dir: Path

    def __post_init__(self):
        # 화면의 네 점 → 한 변이 1인 네모. 먼 쪽이 위(0), 가까운 쪽이 아래(1).
        self._to_unit = cv2.getPerspectiveTransform(self.floor_quad_px.astype(np.float32), UNIT_SQUARE)

    def video(self, clip: str) -> Path:
        return self.video_dir / f"{clip}.mp4"

    def detections(self, clip: str) -> Path:
        return self.detections_dir / f"{clip}.npz"

    def locate(self, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """사람 상자(N×4) → (세는 바닥 안에서의 자리 N×2, 바닥 안인지 N).

        자리는 0~1 사이의 두 숫자다: 왼쪽 0 ~ 오른쪽 1, 먼 쪽 0 ~ 가까운 쪽 1.
        발 위치는 상자의 아래쪽 가운데다 (E-37). 상자가 화면 아래에서 잘리면 화면 끝을 발로 본다.
        """
        b = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if len(b) == 0:
            return np.empty((0, 2)), np.zeros(0, dtype=bool)
        feet = np.stack([(b[:, 0] + b[:, 2]) / 2, np.minimum(b[:, 3], self.image_size[1] - 1)], axis=1)
        unit = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), self._to_unit).reshape(-1, 2)
        inside = (unit[:, 0] >= 0) & (unit[:, 0] <= 1) & (unit[:, 1] >= 0) & (unit[:, 1] <= 1)
        return unit.astype(float), inside


@dataclass
class Zone:
    key: str  # "A", "B", "C"
    name: str
    street: str
    plan_rect: tuple[float, float, float, float]  # 약도에서의 자리 [왼쪽, 위, 가로, 세로]
    area_m2: float | None  # 실제 넓이. 모르면 None (1㎡당 사람 수를 내지 않는다)
    cctv: Cctv

    def to_plan(self, unit: np.ndarray) -> np.ndarray:
        """세는 바닥 안에서의 자리(0~1) → 약도 좌표."""
        u0, v0, w, h = self.plan_rect
        return np.asarray(unit, dtype=float).reshape(-1, 2) * [w, h] + [u0, v0]


@dataclass
class Venue:
    name: str
    plan_size: tuple[float, float]
    roads: list[tuple[float, float, float, float]]  # 길 [왼쪽, 위, 가로, 세로]. 구역 밖의 길은 CCTV가 보지 않는 곳
    clip_seconds: float
    zones: dict[str, Zone]


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def load_venue(path: str | Path) -> Venue:
    cfg = load_yaml(path)
    zones = {}
    for k, z in cfg["zones"].items():
        c = z["cctv"]
        cctv = Cctv(
            id=c["id"], image_size=tuple(c["image_size"]), floor_quad_px=np.array(c["floor_quad_px"], dtype=float),
            clips=list(c["clips"]), video_dir=_path(cfg["video_dir"]), detections_dir=_path(cfg["detections_dir"]),
        )
        zones[k] = Zone(key=k, name=z["name"], street=z["street"], plan_rect=tuple(z["plan_rect"]),
                        area_m2=z.get("area_m2"), cctv=cctv)
    return Venue(name=cfg["name"], plan_size=tuple(cfg["plan_size"]), roads=[tuple(r) for r in cfg.get("roads", [])],
                 clip_seconds=float(cfg["clip_seconds"]), zones=zones)
