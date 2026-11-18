# 이 파일이 하는 일: 사람 상자를 구역 A/B/C별로 세고, 숫자 흔들림을 줄이는 평균을 낸다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-08, C-23, C-30, C-37
# 근거: E-36 (사람 검출 모델), E-37 (발 위치로 구역 판정, supervision PolygonZone), D-08, D-26
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_zones(path: str | Path) -> dict[str, sv.PolygonZone]:
    """구역 선 파일을 읽어 구역 이름별 PolygonZone을 만든다.

    기준점은 상자의 아래쪽 가운데(발 위치)이다. PolygonZone의 기본값과 같다 (E-37).
    """
    cfg = load_yaml(path)
    return {
        name: sv.PolygonZone(
            polygon=np.array(z["polygon"], dtype=np.int64),
            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        )
        for name, z in cfg["zones"].items()
    }


def count_by_zone(xyxy: np.ndarray, zones: dict[str, sv.PolygonZone]) -> dict[str, int]:
    """사람 상자(N×4, x1 y1 x2 y2)를 받아 구역별 사람 수를 돌려준다."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    if len(xyxy) == 0:
        return {name: 0 for name in zones}
    detections = sv.Detections(xyxy=xyxy)
    return {name: int(zone.trigger(detections).sum()) for name, zone in zones.items()}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """지난 window장의 평균. 앞부분은 있는 만큼만 평균한다 (미래 값을 쓰지 않는다)."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    csum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    start = np.maximum(idx - window, 0)
    return (csum[idx] - csum[start]) / (idx - start)
