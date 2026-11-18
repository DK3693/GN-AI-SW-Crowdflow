# 이 파일이 하는 일: 하루 정리와 결과 보고서의 숫자(review.py)가 실행 기록에 있는 것만 세는지 확인한다. AI는 쓰지 않는다.
#   기록은 여기서 손으로 만든 작은 것이다 (실제 실행의 기록이 아님).
# 해당 기능: 핵심 기능 4의 뒤처리 (하루 정리, 결과 보고서), 기억
# 해당 기준: C-32, C-38, C-50
# 근거: D-64, D-75, E-06 (숫자를 지어내지 않음: 숫자는 코드가 기록에서 세고 AI는 글만 씀)
# 출처: AI 코딩 도구로 작성
from crowdflow.agents import RuleBrain
from crowdflow.review import facts_text, field_notes, headline_numbers, incident_rows, report_markdown, result_report_markdown, run_stats


def ev(kind, sec, **more):
    """기록 한 줄. sec: 20:00:00에서 몇 초 지났는지 (영상 시계와 실제 시각을 같게 둔다)."""
    return {"kind": kind, "who": "시험", "what": "", "time": f"2026-10-05T20:{sec // 60:02d}:{sec % 60:02d}", "clock": f"20:{sec // 60:02d}:{sec % 60:02d}", "t": float(sec), **more}


def sample_events():
    return [
        ev("요원 역할 대본", 0, on=True),
        ev("붐빔 감지", 60, incident=1, zone="B", level="주의", count_avg=26.0),
        ev("지침 찾기", 61, plan_id="1-1"),
        ev("토론", 70, plan_id="1-1", who="효율 담당", option=1),
        ev("토론", 70, plan_id="1-1", who="안전 담당", option=2),
        ev("토론", 70, plan_id="1-1", who="지침 담당", option=1),
        ev("토론", 70, plan_id="1-1", who="계획 담당", option=1),
        {**ev("계획", 71, plan_id="1-1", zone="B", chosen=1, situation="상황 글", options_text="선택지 글", query="물음"), "what": "선택지 1 (가까운 근무 요원을 보냄)로 정함"},
        {**ev("계획", 72, plan_id="1-2", zone="B", chosen=1), "what": "선택지가 하나뿐이라 토론 없이 선택지 1로 정함"},  # 토론 없이 정한 것은 세지 않는다
        ev("지시 보냄", 80, zone="B", staff="S5", purpose="붐빔 대응"),
        ev("다친 사람 알림", 85, staff="S5"),
        ev("요원 제외", 85, staff="S5", scripted=True),
        ev("지시 보냄", 86, zone="B", staff="R1", purpose="붐빔 대응"),
        ev("지시 보냄", 87, zone="B", staff="S1", purpose="휴게 교대"),
        ev("도착", 100, staff="R1", scripted=True),
        ev("다시 세기", 120, incident=1, count_avg=27.0),
        ev("결과 확인", 121, incident=1, decision="keep_watching"),
        ev("다시 세기", 140, incident=1, count_avg=18.0),
        ev("결과 확인", 141, incident=1, decision="close"),
        ev("복귀 지시", 142, staff="R1"),
        ev("붐빔 감지", 150, incident=2, zone="C", level="경계", count_avg=30.0),
        ev("관리자에게 넘김", 151, zone="C"),
    ]


def test_run_stats_counts_only_what_is_in_the_record():
    s = run_stats(sample_events(), ["R1", "R2"])
    assert (s["incidents_opened"], s["incidents_closed"]) == (2, 1)
    b, c = s["incidents"]
    assert b["zone"] == "B" and b["sent"] == 2 and b["refusals"] == 1 and b["rechecks"] == 2 and b["closed"]
    assert b["first_order_wait"] == 20 and b["seconds_to_close"] == 81 and (b["before_avg"], b["after_avg"]) == (26.0, 18.0)
    assert c["zone"] == "C" and c["sent"] == 0 and c["first_order_wait"] is None and not c["closed"] and c["no_staff"] == 1
    assert s["orders"] == {"붐빔 대응": 2, "휴게 교대": 1, "최소 인원 채우기": 0}
    assert (s["no_staff"], s["no_staff_zones"], s["injuries"], s["refusals"], s["returns"]) == (1, ["C"], 1, 1, 1)
    assert (s["reserves_total"], s["reserves_used"], s["reserve_orders"], s["reserve_refusals"], s["reserves_all_used_clock"]) == (2, 1, 1, 0, None)
    assert run_stats(sample_events(), ["R1"])["reserves_all_used_clock"] == "20:01:26"  # 예비 요원이 한 명뿐이면 그 한 명이 지시받은 때 다 쓴 것
    assert s["choices"] == {"가까운 근무 요원을 보냄": 1}
    assert (s["debates"], s["debates_split"], s["planner_followed_efficiency"], s["planner_followed_safety"]) == (1, 1, 1, 0)
    assert (s["guideline_reads"], s["guideline_applied"], s["planner_followed_guideline"]) == (1, 1, 1)
    assert s["staff_script"] is True and s["scripted_actions"] == 2  # 대본이 넣은 일은 따로 센다
    assert s["first_order_wait_avg"] == 20.0 and s["seconds_to_close_avg"] == 81.0


def test_empty_record_gives_zeros_not_guesses():
    s = run_stats([], ["R1"])
    assert s["incidents_opened"] == 0 and s["first_order_wait_avg"] is None and s["seconds_to_close_avg"] is None and s["reserves_used"] == 0
    assert "붐빔 사건: 0건" in facts_text(s) and "지침을 읽지 않음" in facts_text(s)


def test_review_text_and_report_carry_the_counted_numbers():
    s = run_stats(sample_events(), ["R1", "R2"])
    facts = facts_text(s)
    for part in ["B구역", "첫 지시까지 20초", "81초 만에 닫음", "관리자에게 넘긴 일: 1번", "2명 가운데 1명", "다친 사람 보고: 1건", "대본이 대신 넣은 것",
                 "'가까운 근무 요원을 보냄' 1번"]:
        assert part in facts
    review = RuleBrain().review(facts, s)
    assert review.staffing == "모자람"  # 보낼 요원이 없던 일이 한 번이라도 있으면 모자람
    text = report_markdown(s, review.model_dump(), "1일차 정리", "규칙")
    for part in ["# 1일차 정리", "녹화 영상", "기록에서 센 숫자", "무엇이 문제였나", "요원 수", "모자람", "관리자가 승인한 것만"]:
        assert part in text


def test_result_report_shows_only_counted_numbers():
    """결과 보고서(D-75)의 큰 숫자, 사건별 표, 현장 기록은 run_stats가 센 값 그대로다."""
    s = run_stats(sample_events(), ["R1", "R2"])
    head = {name: (value, note) for name, value, note in headline_numbers(s)}
    assert head["붐빔 사건"] == ("2건", "그 가운데 1건을 닫음")
    assert head["요원에게 보낸 지시"] == ("3건", "붐빔 대응 2 · 휴게 교대 1 · 인원 채우기 0")
    assert head["도착 뒤 다시 세기"][0] == "2번" and head["붐빔을 잡고 첫 지시까지"][0] == "20.0초"
    assert head["규칙이 AI의 판단을 바로잡은 일"][0] == "0번" and head["관리자에게 넘긴 일"][0] == "1번"
    b, c = incident_rows(s)
    assert (b["zone"], b["opened"], b["wait"], b["sent"], b["rechecks"]) == ("B구역", "20:01:00 · 주의", "20초", "2명 (못 간다고 한 요원 1명)", "2번")
    assert b["closed"] and b["result"] == "81초 만에 닫음 (평균 26.0명 → 18.0명)"
    assert (c["zone"], c["wait"], c["sent"], c["handed"], c["closed"]) == ("C구역", "지시가 나가지 않음", "0명", "1번", False) and "닫지 못함" in c["result"]
    notes = " ".join(field_notes(s))
    for part in ["못 간다고 한 요원 1명", "다친 사람 보고 1건", "휴게 교대 지시 1건", "예비 요원 2명 가운데 1명", "토론 1번", "2건은 사람이 아니라 정해 둔 대본"]:
        assert part in notes
    review = RuleBrain().review(facts_text(s), s).model_dump()
    text = result_report_markdown(s, review, "2026-10-05 21:00:00", "규칙", corrected=True)
    for part in ["# 행사 안전 운영 결과 보고서", "녹화 영상으로 재현", "| B구역 | 20:01:00 · 주의 | 20초 |", "| C구역 |", "## 4. 확인 담당의 정리", "**모자람**",
                 "숫자대로 바로잡음", "정리 글: 규칙"]:
        assert part in text
    assert "지침에 더할 문장" not in text  # 운영 지침(RAG)에 넣을 제안은 결과 보고서에 싣지 않는다 (D-68, D-75)
    assert "정리 글: 확인 담당 (AI)" in result_report_markdown(s, review, "-", "CrewAI 담당")  # AI가 썼으면 그렇게, 규칙이 대신 썼으면 규칙이라고 밝힌다
    empty = run_stats([], ["R1"])
    assert incident_rows(empty) == [] and headline_numbers(empty)[3][1] == "–" and "붐빔 사건이 없었음" in result_report_markdown(empty, review, "-", "규칙")
