# 이 파일이 하는 일: 화면이 쓰는 "실시간 시연"(live.py)을 화면 없이 돌려 본다. 영상 시계를 30배로 빨리 돌리고,
#   사람이 화면에서 할 일(관리자의 글, 요원의 글, 도착)을 대화방을 보면서 넣는다. AI 대신 규칙을 쓴다.
# 해당 기능: 핵심 기능 1~4를 화면에 잇는 부분
# 해당 기준: C-28, C-31, C-32, C-48
# 근거: D-51, D-52, D-53, E-54
# 출처: AI 코딩 도구로 작성
import time
from pathlib import Path

import pytest

from crowdflow import chatroom as chat
from crowdflow.flow import ARRIVAL, MANAGER_TEXT, STAFF_TEXT
from crowdflow.live import LiveOperation
from crowdflow.records import read_events

ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "outputs" / "detections" / "aihub" / "E05_026.npz").exists(),
                                reason="영상 분석 결과가 이 컴퓨터에 없음 (저장소에 올리지 않음)")


def test_live_run_takes_actions_from_screens_and_can_start_over(tmp_path):
    eng = LiveOperation(use_rule=True, out_dir=tmp_path, speed=30)
    try:
        assert not eng.started and eng.t == 0 and eng.clock() == "20:00:00" and eng.model_ready is True
        points, status = eng.view()
        assert set(points) == set(status) == set(eng.venue.zones)
        eng.submit(MANAGER_TEXT, text="A구역은 2명 유지해")  # 시작 전에도 지시를 받는다
        eng.start()
        seen, first, deadline = 0, True, time.time() + 40
        while not eng.finished and time.time() < deadline:
            orders = [m for m in eng.room.all() if m.kind == chat.ORDER]
            for m in orders[seen:]:  # 처음 지시를 받은 요원은 못 간다고 답하고, 그 뒤의 요원은 도착을 누른다
                if first:
                    eng.submit(STAFF_TEXT, staff_id=m.to, text="다친 관람객을 돌보는 중이라 못 갑니다")
                    first = False
                else:
                    eng.submit(ARRIVAL, staff_id=m.to)
            seen = len(orders)
            time.sleep(0.02)
        while eng.waiting and time.time() < deadline:
            time.sleep(0.05)
        assert eng.finished and eng.t == eng.total and eng.waiting == []
        kinds = [e["kind"] for e in read_events(eng.log_path)]
        for name in ["조건 바꿈", "붐빔 감지", "지시 보냄", "요원 제외", "도착"]:
            assert name in kinds
        assert "흐름 오류" not in kinds and eng.op.state.min_staff["A"] == 2
        eng.reset()  # 처음부터 다시: 기록, 대화방, 요원, 조건이 처음 상태로
        assert not eng.started and eng.room.all() == [] and read_events(eng.log_path) == [] and eng.op.state.min_staff["A"] == 1
    finally:
        eng.close()


def test_day_review_counts_the_record_and_approved_sentences_stay_for_the_next_day(tmp_path):
    """하루 정리(D-64): 숫자는 기록에서 세고, 관리자가 승인한 문장만 지침에 들어가 다음 날에도 남는다. "어려운 날" 대본은 못 간다는 요원이 더 많다."""
    eng = LiveOperation(use_rule=True, out_dir=tmp_path, speed=30)
    try:
        base, hard = len(eng.clauses()), eng.settings["scenario"]["staff_script"]["hard_refusals"]
        assert eng.guideline_on is False and eng.hard_day is False and eng.script_refusals() == 1
        eng.staff_script = eng.hard_day = True
        assert eng.script_refusals() == hard
        eng.start()
        deadline = time.time() + 40
        while (not eng.finished or eng.waiting) and time.time() < deadline:
            time.sleep(0.05)
        events = read_events(eng.log_path)
        kinds = [e["kind"] for e in events]
        replies = [e for e in events if e["kind"] == "요원 글"]
        assert eng.finished and "흐름 오류" not in kinds and 1 <= len(replies) <= hard and all(e["scripted"] for e in replies)
        assert "지침 찾기" not in kinds  # 지침 읽기가 꺼져 있으면 찾지 않는다

        item = eng.make_review()
        s = item["stats"]
        assert item["day"] == 1 and item["hard_day"] is True and item["guideline_on"] is False and item["script_refusals"] == hard
        assert s["incidents_opened"] == kinds.count("붐빔 감지") and s["refusals"] == kinds.count("요원 제외") == len(replies)
        assert s["injuries"] == kinds.count("다친 사람 알림") and s["scripted_actions"] == sum(1 for e in events if e.get("scripted"))
        assert len(list((tmp_path / "reports").glob("review_*.md"))) == 1 and eng.reviewed_gen == eng._gen
        assert read_events(eng.log_path)[-1]["kind"] == "하루 정리"

        sentence = "붐빔 대응에는 근무 요원을 먼저 보낸다."
        assert eng.approve([sentence, sentence, " "], "1일차 정리에서 제안 · 관리자 승인") == 1
        assert len(eng.clauses()) == base + 1 and eng.clauses()[-1].id == "L-01" and eng.approve([sentence], "다시") == 0
        eng.reset()  # 다음 날: 기록은 비지만 정리와 승인한 문장은 남는다
        assert read_events(eng.log_path) == [] and len(eng.reviews) == 1 and len(eng.clauses()) == base + 1 and eng.reviewed_gen != eng._gen
        eng.forget()
        assert eng.reviews == [] and len(eng.clauses()) == base and not list((tmp_path / "reports").glob("review_*"))
    finally:
        eng.close()


def test_staff_script_plays_the_staff_role_and_marks_it_in_the_record(tmp_path):
    """요원 역할을 대본이 대신하게 켜면(D-60), 사람이 아무것도 누르지 않아도 답 → 다시 계산 → 도착 → 다시 세기까지 간다."""
    eng = LiveOperation(use_rule=True, out_dir=tmp_path, speed=30)
    try:
        assert eng.staff_script is False  # 기본은 끔: 사람이 요원 화면에서 직접 한다
        eng.staff_script = True
        eng.start()
        deadline = time.time() + 40
        while not eng.finished and time.time() < deadline:
            time.sleep(0.05)
        while eng.waiting and time.time() < deadline:
            time.sleep(0.05)
        events = read_events(eng.log_path)
        kinds = [e["kind"] for e in events]
        assert eng.finished and "흐름 오류" not in kinds
        switched = [e for e in events if e["kind"] == "요원 역할 대본"]
        assert len(switched) == 1 and switched[0]["on"] is True  # 켰다는 것이 기록에 남는다
        replies = [e for e in events if e["kind"] == "요원 글"]
        assert len(replies) == 1 and replies[0]["scripted"] is True and "못 갑니다" in replies[0]["what"]  # "못 갑니다"는 한 번만
        arrivals = [e for e in events if e["kind"] == "도착"]
        assert arrivals and all(e["scripted"] is True for e in arrivals)  # 대본이 누른 도착은 그렇게 적힌다
        for name in ["요원 제외", "다시 세기", "결과 확인"]:
            assert name in kinds
        assert eng.op.state.closed  # 닫힌 사건이 있다 (도착 뒤 다시 세어 기준 아래로 내려간 것을 봄)
    finally:
        eng.close()
