# 이 파일이 하는 일: 구역의 붐빔을 나타내는 값(구역 안 사람 수의 평균)을 붐빔 단계(평상, 주의, 경계, 심각)로 바꾼다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-02, C-24
# 근거: D-45 (구역 안 사람 수로 단계를 나눔, 시연용 기준), D-41 (요원을 보내는 단계). 기준값은 config/settings.yaml에 있다.
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    index: int  # 0 = 가장 낮은 단계
    name: str


def level_for(value: float, cfg: dict) -> Level:
    """붐빔 값 → 단계. cfg는 settings.yaml의 crowd_levels. 값이 min 이상이면 그 단계다."""
    level = Level(0, cfg["below_name"])
    for i, lv in enumerate(sorted(cfg["levels"], key=lambda x: x["min"]), start=1):
        if value >= lv["min"]:
            level = Level(i, lv["name"])
    return level


def needs_dispatch(level: Level, cfg: dict) -> bool:
    """이 단계에서 요원을 보내야 하는가 (D-41)."""
    names = [cfg["below_name"]] + [lv["name"] for lv in sorted(cfg["levels"], key=lambda x: x["min"])]
    return level.index >= names.index(cfg["dispatch_from"])
