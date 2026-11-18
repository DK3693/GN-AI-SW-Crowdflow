# 이 파일이 하는 일: 붐빔 단계 판정과 "요원을 보내는 단계"가 설정 파일의 기준대로 되는지 확인한다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-48 (안정성·시험)
# 근거: D-41, D-45
# 출처: AI 코딩 도구로 작성
from pathlib import Path

import pytest

from crowdflow.crowd_vision import load_yaml
from crowdflow.levels import level_for, needs_dispatch

CFG = load_yaml(Path(__file__).parent.parent / "config" / "settings.yaml")["crowd_levels"]
# 단계를 나누는 규칙을 확인하는 시험이라 기준값은 여기서 고정한다. 시연용 기준(config/settings.yaml)이 바뀌어도 이 시험은 그대로다
CFG["levels"] = [{"name": "주의", "min": 25}, {"name": "경계", "min": 30}, {"name": "심각", "min": 35}]


@pytest.mark.parametrize(
    "people, name, index",
    [(0, "평상", 0), (24.9, "평상", 0), (25, "주의", 1), (29.9, "주의", 1),
     (30, "경계", 2), (34.9, "경계", 2), (35, "심각", 3), (80, "심각", 3)],
)
def test_level_boundaries(people, name, index):
    lv = level_for(people, CFG)
    assert (lv.name, lv.index) == (name, index)


def test_dispatch_starts_at_the_configured_level():
    # D-41: "주의"부터 요원을 보낸다
    assert CFG["dispatch_from"] == "주의"
    assert not needs_dispatch(level_for(24, CFG), CFG)
    assert needs_dispatch(level_for(25, CFG), CFG)
    assert needs_dispatch(level_for(40, CFG), CFG)


def test_levels_work_with_other_thresholds():
    # 기준값이 바뀌어도(예: 1㎡당 사람 수 기준) 같은 코드가 돈다
    cfg = {"below_name": "평상", "levels": [{"name": "경계", "min": 1.08}, {"name": "주의", "min": 0.72}], "dispatch_from": "경계"}
    assert level_for(0.9, cfg).name == "주의" and not needs_dispatch(level_for(0.9, cfg), cfg)
    assert level_for(1.2, cfg).name == "경계" and needs_dispatch(level_for(1.2, cfg), cfg)
