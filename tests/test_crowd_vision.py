# 이 파일이 하는 일: 구역별 세기와 평균 계산이 맞는지 작은 예로 확인한다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-48 (안정성·시험)
# 근거: E-37
# 출처: AI 코딩 도구로 작성
from pathlib import Path

import numpy as np

from crowdflow.crowd_vision import count_by_zone, load_zones, moving_average

ZONES = Path(__file__).parent.parent / "config" / "zones" / "aihub489_ja-ma_C0041.yaml"


def test_foot_position_decides_zone():
    zones = load_zones(ZONES)
    boxes = np.array(
        [
            [100, 700, 200, 1000],  # 발이 y=1000 → A
            [100, 300, 200, 700],  # 머리는 B 띠에 있지만 발이 y=700 → A
            [900, 400, 960, 600],  # 발이 y=600 → B
            [900, 50, 930, 150],  # 발이 y=150 → C
        ]
    )
    assert count_by_zone(boxes, zones) == {"A": 2, "B": 1, "C": 1}


def test_no_people():
    zones = load_zones(ZONES)
    assert count_by_zone(np.empty((0, 4)), zones) == {"A": 0, "B": 0, "C": 0}


def test_moving_average_uses_only_past():
    out = moving_average(np.array([0, 3, 6, 9]), window=2)
    assert out.tolist() == [0.0, 1.5, 4.5, 7.5]
