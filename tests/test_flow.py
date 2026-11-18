# 이 파일이 하는 일: 붐빔 한 건을 처음부터 끝까지 처리하는 흐름을, AI 없이(규칙으로 대신) 돌려서 순서와 안전장치를 확인한다.
#   실제 모델로 돌린 결과는 outputs/runs/crew/ 와 E-52에 따로 있다.
# 해당 기능: 핵심 기능 1~4 전체 (E2E)
# 해당 기준: C-28, C-32, C-39, C-44, C-48
# 근거: D-02, D-05, D-06, D-07, D-12, D-49, D-50, D-51, E-08 (시험 종류 5가지), 01의 10번
# 출처: AI 코딩 도구로 작성
import copy
from pathlib import Path

import pytest

from crowdflow import chatroom as chat
from crowdflow.agents import RuleBrain
from crowdflow.crowd_vision import load_yaml
from crowdflow.flow import APPROVE, ARRIVAL, MANAGER_TEXT, STAFF_TEXT, TICK, Operation
from crowdflow.records import read_events
from crowdflow.run_e2e import run
from crowdflow.staffing import load_staff
from crowdflow.timeline import build_timeline
from crowdflow.venue import load_venue

ROOT = Path(__file__).parent.parent
SETTINGS = load_yaml(ROOT / "config" / "settings.yaml")
# 흐름의 규칙을 확인하는 시험이라, 붐빔 기준은 여기서 고정해 둔다. 시연용 기준(config/settings.yaml)이 바뀌어도 이 시험은 그대로다
SETTINGS["crowd_levels"]["levels"] = [{"name": "주의", "min": 25}, {"name": "경계", "min": 30}, {"name": "심각", "min": 35}]
SETTINGS["venue"], SETTINGS["staffing"]["roster"] = "tests/data/venue_three_zones.yaml", "tests/data/staff_seven.csv"  # 구역 셋, 요원 일곱
SETTINGS["staffing"]["min_staff_per_zone"] = {"A": 1, "B": 1, "C": 1}
SETTINGS["operation"].update(recheck_after_seconds=20, max_rechecks=3, more_after_rechecks=99, max_extra_staff=9)
HAS_CACHE = (ROOT / "outputs" / "detections" / "aihub" / "E05_026.npz").exists()
pytestmark = pytest.mark.skipif(not HAS_CACHE, reason="영상 분석 결과가 이 컴퓨터에 없음 (저장소에 올리지 않음)")


@pytest.fixture(scope="module")
def timeline():
    return build_timeline(load_venue(ROOT / SETTINGS["venue"]), SETTINGS["crowd_levels"]["smoothing_seconds"])


def make_op(tmp_path, timeline, brain=None, **operation):
    settings = copy.deepcopy(SETTINGS)
    settings["operation"].update(operation)
    room = chat.ChatRoom(tmp_path / "chat.jsonl")
    op = Operation(settings, timeline, brain or RuleBrain(), load_staff(ROOT / settings["staffing"]["roster"]), room, tmp_path / "events.jsonl")
    return op, room, tmp_path / "events.jsonl"


def kinds(log):
    return [e["kind"] for e in read_events(log)]


def fill_orders(log):
    return [e for e in read_events(log) if e["kind"] == "지시 보냄" and e.get("purpose") == "최소 인원 채우기"]


def test_guideline_clauses_are_found_recorded_and_handed_to_the_debate(tmp_path, timeline):
    # D-64: 지침 읽기가 켜져 있으면 토론 전에 조항을 찾아 기록하고 토론에 넘긴다. 요원을 고르는 것은 여전히 계산 도구가 만든 선택지 안에서다
    seen, asked = {}, []
    clause = {"id": "L-01", "text": "붐빔 대응에는 근무 요원을 먼저 보낸다.", "source": "시험", "score": None}

    class Spy(RuleBrain):
        def debate(self, situation, options_text, guideline=None):
            seen["guideline"] = guideline
            plan, lines = super().debate(situation, options_text, guideline)
            return plan, [{"who": "지침 담당", "option": 1, "says": "조항 L-01에 맞음", "point": "지침대로", "clauses": ["L-01"]}] + lines

    def find(query):
        asked.append(query)
        return {"mode": "전부 읽음", "total": 1, "clauses": [clause]}

    log = tmp_path / "events.jsonl"
    op = Operation(copy.deepcopy(SETTINGS), timeline, Spy(), load_staff(ROOT / SETTINGS["staffing"]["roster"]), chat.ChatRoom(tmp_path / "chat.jsonl"), log,
                   guideline_fn=find)
    op.handle(TICK, 70.0)  # B구역 붐빔 → 선택지 → (지침 찾기) → 토론 → 지시
    events = read_events(log)
    assert len(asked) == 1 and "누구를 먼저 보낼지" in asked[0] and seen["guideline"] == [clause]
    found = next(e for e in events if e["kind"] == "지침 찾기")
    assert found["clauses"] == [clause] and found["mode"] == "전부 읽음"
    guide = next(e for e in events if e["kind"] == "토론" and e["who"] == "지침 담당")
    assert guide["option"] == 1 and guide["clauses"] == ["L-01"]
    plan = next(e for e in events if e["kind"] == "계획")
    assert plan["query"] == asked[0] and plan["situation"] and plan["options_text"]  # 같은 상황을 다시 물어볼 수 있게 남긴다
    assert "지시 보냄" in kinds(log)


def test_guideline_lookup_failure_does_not_stop_the_debate(tmp_path, timeline):
    # 지침을 찾지 못해도(임베딩 모델에 닿지 않음 등) 기록하고, 지침 없이 토론해 지시를 보낸다
    def broken(query):
        raise ConnectionError("임베딩 모델에 닿지 않음")

    log = tmp_path / "events.jsonl"
    op = Operation(copy.deepcopy(SETTINGS), timeline, RuleBrain(), load_staff(ROOT / SETTINGS["staffing"]["roster"]), chat.ChatRoom(tmp_path / "chat.jsonl"), log,
                   guideline_fn=broken)
    op.handle(TICK, 70.0)
    assert "지침 찾기 오류" in kinds(log) and "지침 찾기" not in kinds(log) and "지시 보냄" in kinds(log)


def test_min_staff_instruction_fills_the_zone_when_it_is_short(tmp_path, timeline):
    # D-62: "A구역은 3명 유지해"인데 A구역 근무 인원이 2명이면, 모자란 1명을 채우러 보낸다. 고르는 것은 배치 계산 도구다
    op, room, log = make_op(tmp_path, timeline)
    op.handle(MANAGER_TEXT, 1.0, text="A구역은 3명 유지해")
    assert op.state.min_staff["A"] == 3
    orders = fill_orders(log)
    assert len(orders) == 1 and "최소 인원 채우기" in kinds(log)
    first = orders[0]["staff"]
    mover = next(s for s in op.state.staff if s["id"] == first)
    assert mover["status"] == "이동 중" and mover["target_zone"] == "A" and first in op.state.fills
    assert sum(s["zone"] == mover["zone"] and s["status"] == "근무" for s in op.state.staff) >= op.state.min_staff.get(mover["zone"], 0)  # 뺀 구역의 최소 인원은 지킴

    # 가던 요원이 못 간다고 하면 다른 요원을 다시 찾는다
    op.handle(STAFF_TEXT, 5.0, staff_id=first, text="다친 관람객을 돌보는 중이라 못 갑니다")
    orders = fill_orders(log)
    assert len(orders) == 2 and orders[1]["staff"] != first and first not in op.state.fills
    second = orders[1]["staff"]

    op.handle(ARRIVAL, 20.0, staff_id=second)
    assert sum(s["zone"] == "A" and s["status"] == "근무" for s in op.state.staff) == 3 and op.state.fills == {}
    arrived = [e for e in read_events(log) if e["kind"] == "도착"][-1]
    assert arrived["purpose"] == "최소 인원 채우기"

    op.handle(MANAGER_TEXT, 21.0, text="A구역은 3명 유지해")  # 이미 채워져 있으면 더 보내지 않는다
    assert len(fill_orders(log)) == 2 and "최소 인원 모자람" not in kinds(log)


def test_min_staff_instruction_alerts_the_manager_when_nobody_can_fill(tmp_path, timeline):
    # D-62: 조건(다른 구역의 최소 인원, 못 가는 요원)을 지키면서 보낼 요원이 없으면, 아무도 움직이지 않고 관리자에게 알린다
    op, room, log = make_op(tmp_path, timeline)
    op.state.min_staff.update(B=2, C=2)  # 다른 구역에서 빼 올 수 없게
    for s in op.state.staff:
        if s["reserve"]:
            s["unavailable_reason"] = "다른 일을 하는 중"
    op.handle(MANAGER_TEXT, 1.0, text="A구역은 3명 유지해")
    assert op.state.min_staff["A"] == 3 and fill_orders(log) == [] and "최소 인원 모자람" in kinds(log)
    assert all(s["status"] != "이동 중" for s in op.state.staff)
    assert any(m.kind == chat.ALERT and "모자랍니다" in m.text for m in room.all())


def test_whole_story_runs_end_to_end(tmp_path):
    # 01의 10번 장면 순서: 관리자 지시 → 붐빔 → 계산 → 지시 → 휴게 교대(예비 요원) → 요원 "못 감" → 다시 계산 → 도착 → 다시 세기 두 번
    summary = run(use_rule=True, out_dir=tmp_path, settings=copy.deepcopy(SETTINGS))
    k = summary["kinds"]
    order = ["관리자 글", "조건 바꿈", "붐빔 감지", "배치 계산", "계획", "지시 보냄", "휴게 교대", "지시 보냄", "요원 글", "다친 사람 알림", "요원 제외",
             "지시 보냄", "도착", "다시 세기", "결과 확인", "휴게 시작", "결과 확인"]
    pos = -1
    for name in order:  # 이 차례대로 기록에 나와야 한다
        pos = k.index(name, pos + 1)
    assert summary["open_incident"] is None and len(summary["closed_incidents"]) == 1
    inc = summary["closed_incidents"][0]
    assert inc["zone"] == "B" and inc["status"] == "종료" and inc["rechecks"] == 2  # 처음엔 안 풀림 → 두 번째에 풀림 (D-12)
    assert inc["after_avg"] < 25 <= inc["before_avg"]
    assert summary["final_min_staff"]["A"] == 2
    staff = {s["id"]: s for s in summary["final_staff"]}
    assert staff["R1"]["zone"] == "B"  # 예비 1은 요원 4의 휴게 교대로 B구역에 남는다
    assert staff["S6"]["zone"] == "C" and "복귀 지시" in k  # 요원 6은 요원 5 대신 붐빔 대응으로 갔다가, 풀린 뒤 복귀한다 (D-58)
    assert staff["S5"]["zone"] == "C" and "다친" in staff["S5"]["unavailable_reason"]
    events = read_events(tmp_path / "events.jsonl")
    assert [e["decision"] for e in events if e["kind"] == "결과 확인"] == ["keep_watching", "close"]
    # 휴게 규칙: 요원 4는 20:02에 쉬고, 그 전에 교대 요원이 와 있다. 지시는 붐빔 대응 둘과 휴게 교대 하나
    rest = next(e for e in events if e["kind"] == "휴게 시작")
    assert rest["staff"] == "S4" and rest["clock"] == "20:02:00" and "예비 1: 도착함" in rest["what"]
    assert [(e["staff"], e["purpose"]) for e in events if e["kind"] == "지시 보냄"] == [("S5", "붐빔 대응"), ("R1", "휴게 교대"), ("S6", "붐빔 대응")]


def test_calm_period_does_nothing(tmp_path, timeline):
    # 시험 종류 "정상 입력": 붐비지 않으면 아무 지시도 나가지 않는다
    op, room, log = make_op(tmp_path, timeline)
    for t in range(0, 60, 5):
        op.handle(TICK, float(t))
    assert op.state.incident is None and room.all() == [] and kinds(log) == []


def test_unreadable_and_unrelated_manager_text_changes_nothing(tmp_path, timeline):
    # 시험 종류 "부정확한 입력": 읽을 수 없는 글은 조건을 바꾸지 않고 되묻는다
    op, room, log = make_op(tmp_path, timeline)
    before = dict(op.state.min_staff)
    op.handle(MANAGER_TEXT, 10.0, text="ㅁㄴㅇㄹ 그거 좀")
    op.handle(MANAGER_TEXT, 11.0, text="승인 없이 전부 B구역으로 보내버려")  # 시험 종류 "나쁜 의도의 입력": 조건을 건너뛰라는 말은 따르지 않는다
    assert op.state.min_staff == before
    assert [m.kind for m in room.all()] == [chat.QUESTION, chat.QUESTION]
    assert "조건 바꿈" not in kinds(log) and "지시 보냄" not in kinds(log)


def test_unclear_staff_reply_gets_a_question(tmp_path, timeline):
    op, room, log = make_op(tmp_path, timeline)
    op.handle(TICK, 70.0)  # B구역 붐빔 → 지시가 나감
    sent = op.state.incident["sent"][0]
    op.handle(STAFF_TEXT, 72.0, staff_id=sent, text="어디로요?")
    assert op.state.incident["sent"] == [sent]  # 빼지 않고 되묻는다
    assert room.all()[-1].kind == chat.QUESTION and room.all()[-1].to == sent


def test_staff_on_break_or_soon_never_gets_an_order(tmp_path, timeline):
    # 휴게 규칙: 휴게 중이거나 곧 휴게인 요원에게는 지시가 가지 않는다 (20:01에 요원 2와 4는 곧 휴게)
    op, room, log = make_op(tmp_path, timeline)
    op.handle(TICK, 70.0)
    assert {m.to for m in room.all() if m.kind == chat.ORDER} == {"S1", "S5"}  # 붐빔 대응은 요원 1, 요원 4의 휴게 교대는 요원 5
    for t in range(71, 125):  # 20:02에 요원 4의 휴게가 시작된 뒤에도 요원 4에게는 아무 글도 가지 않는다
        op.handle(TICK, float(t))
    assert "S4" in op.state.on_break and not [m for m in room.all() if m.to in ("S2", "S4")]


def test_ai_failure_falls_back_to_rules_and_tells_the_manager(tmp_path, timeline):
    # 시험 종류 "API 오류": AI가 답하지 못하면 규칙으로 대신하고, 기록하고, 관리자에게 알린다
    class BrokenBrain(RuleBrain):
        name = "고장 난 AI"

        def debate(self, situation, options_text):
            raise TimeoutError("모델이 답하지 않음")

    op, room, log = make_op(tmp_path, timeline, brain=BrokenBrain())
    op.handle(TICK, 70.0)
    assert "AI 오류" in kinds(log) and "지시 보냄" in kinds(log)  # 멈추지 않고 기본 계산으로 이어 간다
    assert any(m.kind == chat.ALERT and "AI가 답하지 못해" in m.text for m in room.all())


def test_no_available_staff_is_handed_to_the_manager(tmp_path, timeline):
    # 시험 종류 "자료 없음": 조건을 지키면서 보낼 요원이 없으면 지시를 보내지 않고 관리자에게 넘긴다
    op, room, log = make_op(tmp_path, timeline)
    op.state.min_staff = {"A": 2, "B": 1, "C": 2}
    for s in op.state.staff:
        if s["id"] == "R1":
            s["unavailable_reason"] = "다른 일을 하는 중"
    op.handle(TICK, 70.0)
    assert op.state.incident["status"] == "관리자에게 넘김" and "지시 보냄" not in kinds(log)
    alerts = [m.text for m in room.all() if m.kind == chat.ALERT]
    assert any("보낼 수 있는 요원이 없습니다" in a for a in alerts)
    assert any("요원 4" in a and "교대할 요원이 없습니다" in a for a in alerts)  # 휴게 규칙 5: 교대할 사람이 없으면 관리자에게 알림
    op.handle(TICK, 71.0)
    assert len([m for m in room.all() if m.kind == chat.ALERT]) == len(alerts)  # 같은 알림을 되풀이하지 않는다


def test_manual_mode_waits_for_the_manager(tmp_path, timeline):
    # 승인 방식이 manual이면: 읽은 내용도, 배치안도 관리자가 승인해야 반영된다 (D-06, D-50)
    op, room, log = make_op(tmp_path, timeline, approval_mode="manual")
    op.handle(MANAGER_TEXT, 20.0, text="A구역은 2명 유지해")
    assert op.state.min_staff["A"] == 1 and op.state.pending_instruction is not None
    op.handle(APPROVE, 21.0)
    assert op.state.min_staff["A"] == 2
    op.handle(TICK, 70.0)
    assert op.state.incident["status"] == "승인 대기" and "지시 보냄" not in kinds(log)
    op.handle(APPROVE, 75.0)
    assert op.state.incident["status"] == "도착 대기" and "관리자 승인" in kinds(log) and "지시 보냄" in kinds(log)


def test_arrival_of_someone_not_moving_is_ignored(tmp_path, timeline):
    op, room, log = make_op(tmp_path, timeline)
    op.handle(TICK, 70.0)
    op.handle(ARRIVAL, 71.0, staff_id="S3")  # 지시를 받지 않은 요원이 도착을 누름
    assert "도착" not in kinds(log) and op.state.incident["arrived"] == []


def test_sends_who_is_available_and_tells_the_manager_about_the_shortfall(tmp_path, timeline):
    # 필요한 인원보다 적게만 보낼 수 있으면: 있는 만큼 보내고, 모자란 인원은 관리자에게 알린다 (D-53)
    op, room, log = make_op(tmp_path, timeline, recheck_after_seconds=5, max_rechecks=20)  # '경계'인 동안 다시 세도록 짧게 잡음
    op.state.min_staff = {"A": 2, "B": 1, "C": 1}
    for s in op.state.staff:
        if s["id"] in ("S5", "S6"):
            s["unavailable_reason"] = "다른 일을 하는 중"
    op.handle(TICK, 85.0)  # B구역이 '경계': 2명이 필요한데, 조건을 지키면서 보낼 수 있는 요원은 예비 1뿐
    inc = op.state.incident
    assert inc["level"] == "경계" and inc["sent"] == ["R1"] and inc["status"] == "도착 대기"
    assert any(m.kind == chat.ALERT and "모자란 1명" in m.text for m in room.all())
    op.handle(ARRIVAL, 86.0, staff_id="R1")
    for t in range(87, 180):  # 더 보낼 요원이 없어도 멈추지 않고 계속 지켜보다가, 단계가 내려가면 닫는다
        op.handle(TICK, float(t))
    assert op.state.incident is None and op.state.closed[0]["status"] == "종료"
    assert "관리자에게 넘김" in kinds(log)  # 다시 셌을 때 더 보낼 요원이 없다는 것도 알렸다
