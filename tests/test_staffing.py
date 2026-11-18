# 이 파일이 하는 일: 요원 배치 계산이 조건을 지키는지, 고른 이유와 뺀 이유가 맞게 나오는지 정해진 상황들로 확인한다.
# 해당 기능: 핵심 기능 3 (요원 배치 계산)
# 해당 기준: C-48 (안정성·시험), C-44 (안전장치)
# 근거: D-08, D-09, D-47, 01의 10번 (시연 장면 순서)
# 출처: AI 코딩 도구로 작성
from pathlib import Path

import pytest

from crowdflow.crowd_vision import load_yaml
from crowdflow.records import log_event, read_events
from crowdflow.staffing import (
    Staff, arrive, clock_to_minutes, extra_needed, load_staff, minutes_until_break, on_break, on_duty_count,
    plan_dispatch, plan_relief, start_move,
)

ROOT = Path(__file__).parent.parent
CFG = load_yaml(ROOT / "config" / "settings.yaml")["staffing"]
NOW = clock_to_minutes("20:01")  # 시연에서 B구역이 붐비기 시작하는 때 (영상 65초쯤, 시작 시각 20:00)
# 계산 규칙을 확인하려고 고정해 둔 요원들. 시연용 자료(config/staff.csv)의 휴게 시각이 바뀌어도 이 시험은 그대로다
ROSTER = [("S1", "요원 1", "A", "20:40", False), ("S2", "요원 2", "A", "20:05", False), ("S3", "요원 3", "B", "20:50", False),
          ("S4", "요원 4", "B", "20:30", False), ("S5", "요원 5", "C", "20:45", False), ("S6", "요원 6", "C", "20:08", False),
          ("R1", "예비 1", CFG["standby_zone"], "20:35", True)]


@pytest.fixture
def staff():
    return [Staff(id=i, name=n, zone=z, reserve=r, next_break=clock_to_minutes(b)) for i, n, z, b, r in ROSTER]


def by_id(staff, i):
    return next(s for s in staff if s.id == i)


def test_roster_matches_the_demo_scale():
    # 처음(D-11)은 요원 6명과 예비 1명, D-58에서 구역이 넷이 되어 요원 8명, D-61에서 예비 요원을 5명으로 늘림. 구역마다 2명씩 (D-47)
    staff = load_staff(ROOT / CFG["roster"])
    assert sum(not s.reserve for s in staff) == 8 and sum(s.reserve for s in staff) == 5
    assert {z: sum(s.zone == z and not s.reserve for s in staff) for z in "ABCD"} == {"A": 2, "B": 2, "C": 2, "D": 2}
    assert all(s.zone == CFG["standby_zone"] for s in staff if s.reserve)


def test_break_clock():
    s = load_staff(ROOT / CFG["roster"])[1]  # 요원 2: 휴게 20:05
    assert minutes_until_break(s, clock_to_minutes("20:01")) == 4
    assert not on_break(s, clock_to_minutes("20:04"), CFG)
    assert on_break(s, clock_to_minutes("20:05"), CFG) and on_break(s, clock_to_minutes("20:14"), CFG)
    assert not on_break(s, clock_to_minutes("20:15"), CFG)
    assert on_break(s, clock_to_minutes("21:07"), CFG)  # 1시간마다 되풀이 (D-09)


def test_needed_staff_follows_the_level_table():
    assert [extra_needed(n, CFG) for n in ("평상", "주의", "경계", "심각")] == [0, 1, 2, 3]  # D-47의 3번


def test_scene_4_first_plan_picks_the_nearest_eligible(staff):
    # 01의 10번 장면 4: B구역이 주의 → 1명 필요. 요원 2와 6은 곧 휴게라 빠지고, 가까운 요원 중 번호가 앞선 요원 1이 뽑힌다
    plan = plan_dispatch("B", 1, staff, NOW, CFG, dict(CFG["min_staff_per_zone"]))
    assert plan.solved and [m.staff_id for m in plan.moves] == ["S1"]
    assert plan.moves[0].travel == CFG["travel_seconds"]["A-B"]
    reasons = {j.staff_id: j.reason for j in plan.excluded}
    assert "곧 휴게" in reasons["S2"] and "곧 휴게" in reasons["S6"]
    assert "이미 채워짐" in reasons["S5"] and "이미 채워짐" in reasons["R1"]
    assert "S3" not in reasons and "S4" not in reasons  # 이미 B구역에 있는 요원은 따지지 않는다


def test_scene_5_manager_raises_minimum_and_nearest_drops_out(staff):
    # 장면 5~6: 관리자가 "A구역은 2명 유지해" → 요원 1이 빠지고 요원 5가 뽑힌다
    min_staff = dict(CFG["min_staff_per_zone"], A=2)
    plan = plan_dispatch("B", 1, staff, NOW, CFG, min_staff)
    assert [m.staff_id for m in plan.moves] == ["S5"]
    assert "최소 인원(2명)이 깨짐" in {j.staff_id: j.reason for j in plan.excluded}["S1"]


def test_scene_8_staff_cannot_go_and_reserve_is_chosen(staff):
    # 장면 8~10: 요원 5가 "다친 사람을 돌보는 중이라 못 가요" → 요원 5를 빼고 다시 계산 → 예비 요원
    min_staff = dict(CFG["min_staff_per_zone"], A=2)
    by_id(staff, "S5").unavailable_reason = "다친 관람객을 돌보는 중"
    plan = plan_dispatch("B", 1, staff, NOW, CFG, min_staff)
    assert plan.solved and [m.staff_id for m in plan.moves] == ["R1"]
    assert "예비 요원" in plan.moves[0].reason
    reasons = {j.staff_id: j.reason for j in plan.excluded}
    assert "못 감" in reasons["S5"] and "다친 관람객" in reasons["S5"]
    text = plan.summary()
    assert "선택: 예비 1" in text and "제외: 요원 5" in text and "제외: 요원 1" in text


def test_no_answer_when_nobody_can_go(staff):
    # 조건을 다 지키는 답이 없으면 "답 없음"과 이유를 낸다
    min_staff = dict(CFG["min_staff_per_zone"], A=2)
    by_id(staff, "S5").unavailable_reason = "다친 관람객을 돌보는 중"
    start_move(by_id(staff, "R1"), "C")  # 예비 요원은 이미 다른 곳으로 이동 중
    plan = plan_dispatch("B", 1, staff, NOW, CFG, min_staff)
    assert not plan.solved and plan.shortfall == 1 and plan.moves == []
    assert "이동 중" in {j.staff_id: j.reason for j in plan.excluded}["R1"]
    assert "답 없음" in plan.summary()


def test_staff_on_break_is_never_sent(staff):
    now = clock_to_minutes("20:06")  # 요원 2가 휴게 중 (20:05~20:15)
    plan = plan_dispatch("B", 3, staff, now, CFG, {"A": 0, "B": 0, "C": 0})
    reasons = {j.staff_id: j.reason for j in plan.excluded}
    assert "휴게 중" in reasons["S2"]
    assert "S2" not in [m.staff_id for m in plan.moves]


def test_minimum_staffing_counts_people_already_taken(staff):
    # 한 구역에서 여러 명을 한꺼번에 빼서 최소 인원이 깨지면 안 된다
    now = clock_to_minutes("20:20")  # 곧 휴게인 요원이 적은 때 (요원 4만 20:30)
    plan = plan_dispatch("B", 3, staff, now, CFG, dict(CFG["min_staff_per_zone"]))
    after = {z: on_duty_count(staff, z, now, CFG) - sum(m.zone == z for m in plan.moves) for z in "AC"}
    assert all(after[z] >= CFG["min_staff_per_zone"][z] for z in "AC")
    assert len(plan.moves) == 3 and plan.moves[-1].staff_id == "R1"  # 정규 요원 2명 뒤에 예비 요원


def test_two_levels_need_two_people_and_one_person_one_task(staff):
    now = clock_to_minutes("20:20")
    first = plan_dispatch("B", extra_needed("주의", CFG), staff, now, CFG, dict(CFG["min_staff_per_zone"]))
    start_move(by_id(staff, first.moves[0].staff_id), "B")
    # 경계로 오르면 2명이 필요하고, 이미 1명이 가고 있으므로 1명만 더 고른다. 가고 있는 요원을 또 고르지 않는다
    second = plan_dispatch("B", extra_needed("경계", CFG) - 1, staff, now, CFG, dict(CFG["min_staff_per_zone"]))
    assert len(second.moves) == 1 and second.moves[0].staff_id != first.moves[0].staff_id
    moved = by_id(staff, first.moves[0].staff_id)
    arrive(moved)
    assert moved.zone == "B" and moved.status == "근무" and moved.target_zone is None


def test_relief_for_staff_about_to_rest_in_a_crowded_zone(staff):
    # 휴게 규칙 4번: 붐비는 곳의 요원이 곧 쉬면 바꿔 줄 요원을 찾는다. 휴게는 미루지 않는다
    now = clock_to_minutes("20:22")  # 요원 4(B구역)가 20:30에 휴게 → 8분 뒤
    plans = plan_relief(staff, ["B"], now, CFG, dict(CFG["min_staff_per_zone"]))
    assert len(plans) == 1 and "휴게 교대" in plans[0].purpose and "요원 4" in plans[0].purpose
    assert plans[0].solved and len(plans[0].moves) == 1
    assert plan_relief(staff, ["A"], clock_to_minutes("20:25"), CFG, dict(CFG["min_staff_per_zone"])) == []


def test_event_log_keeps_what_and_why(tmp_path):
    log = tmp_path / "events.jsonl"
    log_event("배치 계산", "배치 계산 도구", "B구역에 요원 1을 보내는 안", why="주의 단계, 1명 필요", path=log, zone="B", staff=["S1"])
    log_event("관리자 승인", "관리자", "승인함", path=log)
    events = read_events(log)
    assert [e["kind"] for e in events] == ["배치 계산", "관리자 승인"]
    assert events[0]["why"] == "주의 단계, 1명 필요" and events[0]["staff"] == ["S1"] and "time" in events[0]
