# 이 파일이 하는 일: 하루치(시연 한 번) 운영 기록을 정리할 재료를 만든다.
#   숫자는 여기서 실행 기록(events.jsonl)에서 센다. 기록에 없는 것은 세지 않는다.
#   글(무엇이 문제였나, 무엇이 더 필요했나, 요원 수가 적당했나, 지침에 더할 문장)은 확인 담당(AI)이 이 숫자만 보고 쓴다 (agents.py의 review).
#   같은 숫자와 글을 "결과 보고서" 꼴(큰 숫자, 사건별 표, 현장에서 들어온 일, 확인 담당의 정리)로 내는 것도 여기서 한다.
# 해당 기능: 핵심 기능 4 (실행과 재확인)의 뒤처리. 기억(기록을 다음 날에 쓰기)
# 해당 기준: C-10, C-25, C-38, C-39, C-43, C-50
# 근거: D-64 (하루를 마치면 정리하고, 승인한 문장을 지침에 더함. 사용자 지시 2026-10-05), D-17 (효과는 걸린 시간 등으로 잼),
#   01의 12번 (사건 기록을 저장), E-06 (숫자를 지어내지 않음: 숫자는 코드가 기록에서 세고 AI는 글만 씀),
#   D-75 (실행이 끝나면 결과 보고서 꼴로 보여 줌. 사용자 지시 2026-10-05)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import re
from datetime import datetime

CROWD, RELIEF, FILL = "붐빔 대응", "휴게 교대", "최소 인원 채우기"  # flow.py와 같은 글자 (기록에 이 글자로 남는다)
DECISIONS = {"close": "사건을 닫음", "send_more": "요원을 더 보냄", "keep_watching": "더 지켜봄"}


def _at(e: dict) -> datetime:
    return datetime.fromisoformat(e["time"])


def run_stats(events: list[dict], reserve_ids: list[str] | tuple[str, ...] = ()) -> dict:
    """기록에서 센 숫자들. reserve_ids: 예비 요원의 번호 (예비 요원을 몇 명 썼는지 세는 데 쓴다)."""
    incidents: list[dict] = []
    open_by_zone: dict[str, dict] = {}
    by_id: dict[int, dict] = {}
    last_order_zone: dict[str, tuple[str, str]] = {}  # 요원 → (가장 최근에 지시받은 구역, 까닭)
    orders = {CROWD: 0, RELIEF: 0, FILL: 0}
    reserves_out: set[str] = set()
    debates: dict[str, dict] = {}
    counts = {"no_staff": 0, "no_relief": 0, "fill_short": 0, "injuries": 0, "refusals": 0, "corrections": 0, "ai_errors": 0,
              "breaks_started": 0, "returns": 0, "manager_alerts": 0, "cancels": 0, "scripted_actions": 0, "min_staff_changes": 0}
    no_staff_zones: list[str] = []
    script_on = False
    choices: dict[str, int] = {}  # 토론으로 정한 결정에서 고른 선택지의 제목 → 횟수
    reserve_orders = reserve_refusals = 0
    all_used_clock = None  # 마지막 예비 요원까지 지시를 받은 시각 (그 뒤로는 남은 예비 요원이 없다)
    for e in events:
        k, zone = e["kind"], e.get("zone")
        if e.get("scripted"):
            counts["scripted_actions"] += 1
        if k == "요원 역할 대본":
            script_on = script_on or bool(e.get("on"))
        elif k == "붐빔 감지":
            inc = {"id": e.get("incident"), "zone": zone, "level": e.get("level"), "opened_clock": e.get("clock"), "opened_t": e.get("t"),
                   "opened_at": _at(e), "before_avg": e.get("count_avg"), "sent": 0, "first_order_wait": None, "refusals": 0, "rechecks": 0,
                   "decisions": [], "more_sent": 0, "no_staff": 0, "closed": False, "closed_clock": None, "seconds_to_close": None, "after_avg": None}
            incidents.append(inc)
            open_by_zone[zone] = inc
            by_id[inc["id"]] = inc
        elif k == "지시 보냄":
            purpose = e.get("purpose")
            if purpose in orders:
                orders[purpose] += 1
            if e.get("staff") in reserve_ids:
                reserves_out.add(e["staff"])
                reserve_orders += 1
                if all_used_clock is None and len(reserves_out) == len(set(reserve_ids)):
                    all_used_clock = e.get("clock")
            last_order_zone[e.get("staff", "")] = (zone, purpose)
            inc = open_by_zone.get(zone)
            if inc and purpose == CROWD:
                inc["sent"] += 1
                if inc["first_order_wait"] is None:
                    inc["first_order_wait"] = int((_at(e) - inc["opened_at"]).total_seconds())
                if inc["rechecks"] > 0:
                    inc["more_sent"] += 1
        elif k == "요원 제외":
            counts["refusals"] += 1
            reserve_refusals += e.get("staff") in reserve_ids
            z, purpose = last_order_zone.get(e.get("staff", ""), (None, None))
            if purpose == CROWD and z in open_by_zone:
                open_by_zone[z]["refusals"] += 1
        elif k == "다친 사람 알림":
            counts["injuries"] += 1
        elif k == "다시 세기":
            inc = by_id.get(e.get("incident"))
            if inc:
                inc["rechecks"] += 1
                inc["after_avg"] = e.get("count_avg")
        elif k == "결과 확인":
            inc = by_id.get(e.get("incident"))
            if inc:
                inc["decisions"].append(e.get("decision"))
                if e.get("decision") == "close":
                    inc["closed"], inc["closed_clock"] = True, e.get("clock")
                    inc["seconds_to_close"] = round(e.get("t", 0) - (inc["opened_t"] or 0))
                    open_by_zone.pop(inc["zone"], None)
        elif k == "관리자에게 넘김":
            counts["no_staff"] += 1
            no_staff_zones.append(zone)
            if zone in open_by_zone:
                open_by_zone[zone]["no_staff"] += 1
        elif k == "교대 요원 없음":
            counts["no_relief"] += 1
        elif k == "최소 인원 모자람":
            counts["fill_short"] += 1
        elif k == "AI 답 바로잡음":
            counts["corrections"] += 1
        elif k == "AI 오류":
            counts["ai_errors"] += 1
        elif k == "휴게 시작":
            counts["breaks_started"] += 1
        elif k == "복귀 지시":
            counts["returns"] += 1
        elif k == "관리자에게 알림":
            counts["manager_alerts"] += 1
        elif k == "지시 취소":
            counts["cancels"] += 1
        elif k == "조건 바꿈":
            counts["min_staff_changes"] += 1
        elif k == "토론":
            debates.setdefault(e.get("plan_id", ""), {})[e["who"]] = e.get("option")
        elif k == "계획" and e.get("situation"):  # 토론으로 정한 결정: "선택지 2 (예비 요원을 먼저 보냄)로 정함"에서 제목을 꺼낸다
            m = re.search(r"\((.+)\)로 정함", e.get("what", ""))
            if m:
                choices[m.group(1)] = choices.get(m.group(1), 0) + 1
    for inc in incidents:
        inc.pop("opened_at")
    full = [d for d in debates.values() if {"효율 담당", "안전 담당", "계획 담당"} <= set(d)]
    split = [d for d in full if d["효율 담당"] != d["안전 담당"]]
    guided = [d for d in full if d.get("지침 담당")]  # 지침 담당이 해당하는 조항을 찾아 선택지를 말한 토론
    waits = [i["first_order_wait"] for i in incidents if i["first_order_wait"] is not None]
    closes = [i["seconds_to_close"] for i in incidents if i["seconds_to_close"] is not None]
    return {
        "events": len(events),
        "video_seconds": round(max((e.get("t", 0) for e in events), default=0)),
        "incidents": incidents,
        "incidents_opened": len(incidents), "incidents_closed": sum(i["closed"] for i in incidents),
        "orders": orders,
        "first_order_wait_avg": round(sum(waits) / len(waits), 1) if waits else None,
        "seconds_to_close_avg": round(sum(closes) / len(closes), 1) if closes else None,
        "more_sent": sum(i["more_sent"] for i in incidents),
        "no_staff_zones": no_staff_zones,
        "reserves_total": len(reserve_ids), "reserves_used": len(reserves_out),
        "reserve_orders": reserve_orders, "reserve_refusals": reserve_refusals, "reserves_all_used_clock": all_used_clock, "choices": choices,
        "debates": len(full), "debates_split": len(split),
        "planner_followed_efficiency": sum(d["계획 담당"] == d["효율 담당"] != d["안전 담당"] for d in full),
        "planner_followed_safety": sum(d["계획 담당"] == d["안전 담당"] != d["효율 담당"] for d in full),
        "guideline_reads": sum(1 for e in events if e["kind"] == "지침 찾기"),
        "guideline_applied": len(guided), "planner_followed_guideline": sum(d["계획 담당"] == d["지침 담당"] for d in guided),
        "staff_script": script_on,
        **counts,
    }


def facts_text(s: dict) -> str:
    """AI에게 넘길 글. 숫자는 모두 run_stats가 기록에서 센 것이다."""
    lines = [f"오늘 운영 기록에서 센 숫자 (기록 {s['events']}건, 마지막 기록은 영상 {s['video_seconds']}초):",
             f"- 붐빔 사건: {s['incidents_opened']}건이 열리고 {s['incidents_closed']}건이 닫힘."]
    for i in s["incidents"]:
        wait = "지시가 나가지 않음" if i["first_order_wait"] is None else f"붐빔을 잡은 뒤 첫 지시까지 {i['first_order_wait']}초"
        end = f"{i['seconds_to_close']}초 만에 닫음(평균 {i['before_avg']}명 → {i['after_avg']}명)" if i["closed"] else "영상이 끝날 때까지 닫지 못함"
        lines.append(f"  · {i['zone']}구역 {i['opened_clock']} '{i['level']}': {wait}, 요원 {i['sent']}명에게 지시(그중 안 풀려서 더 보낸 요원 {i['more_sent']}명), "
                     f"못 간다고 한 요원 {i['refusals']}명, 다시 세기 {i['rechecks']}번, 보낼 요원이 없던 적 {i['no_staff']}번, {end}.")
    lines += [
        f"- 지시: 붐빔 대응 {s['orders'][CROWD]}건, 휴게 교대 {s['orders'][RELIEF]}건, 최소 인원 채우기 {s['orders'][FILL]}건. 사건을 닫고 돌려보낸 일 {s['returns']}번.",
        f"- 조건을 지키면서 보낼 요원이 없어 관리자에게 넘긴 일: {s['no_staff']}번" + (f" ({', '.join(z + '구역' for z in s['no_staff_zones'])})." if s["no_staff_zones"] else "."),
        f"- 휴게 교대 요원을 구하지 못한 일: {s['no_relief']}번. 관리자가 정한 최소 인원을 채우지 못한 일: {s['fill_short']}번.",
        f"- 예비 요원: {s['reserves_total']}명 가운데 {s['reserves_used']}명이 지시를 받음 (예비 요원에게 간 지시 {s['reserve_orders']}건, 못 간다고 한 예비 요원 {s['reserve_refusals']}명)."
        + (f" {s['reserves_all_used_clock']}에 마지막 예비 요원까지 지시를 받아, 그 뒤로는 새로 쓸 예비 요원이 남아 있지 않았음." if s["reserves_all_used_clock"] else ""),
        f"- 다친 사람 보고: {s['injuries']}건. 현장 사정으로 못 간다고 한 요원: {s['refusals']}명.",
        f"- 휴게: 시작 {s['breaks_started']}건. 관리자의 최소 인원 지시: {s['min_staff_changes']}건. 관리자가 취소한 지시: {s['cancels']}건.",
        f"- 토론 {s['debates']}번 가운데 효율 담당과 안전 담당의 선택이 갈린 것 {s['debates_split']}번 "
        f"(계획 담당이 효율 쪽을 따름 {s['planner_followed_efficiency']}번, 안전 쪽을 따름 {s['planner_followed_safety']}번).",
        f"- AI의 판단을 규칙이 바로잡은 일: {s['corrections']}번. AI가 답하지 못해 규칙으로 대신한 일: {s['ai_errors']}번.",
    ]
    if s["choices"]:
        lines.insert(-1, "- 토론에서 계획 담당이 고른 선택지: " + ", ".join(f"'{t}' {n}번" for t, n in s["choices"].items()) + ".")
    if s["guideline_reads"]:
        lines.append(f"- 운영 지침: 토론 {s['guideline_reads']}번에서 지침을 읽음. 해당하는 조항이 있던 것 {s['guideline_applied']}번, "
                     f"그 가운데 계획 담당이 조항대로 정한 것 {s['planner_followed_guideline']}번.")
    else:
        lines.append("- 운영 지침: 오늘은 토론에서 지침을 읽지 않음.")
    if s["staff_script"] or s["scripted_actions"]:
        lines.append(f"- 참고: 요원의 답과 도착 가운데 {s['scripted_actions']}건은 대본이 대신 넣은 것임 (사람이 아니라 정해 둔 대본).")
    return "\n".join(lines)


def report_markdown(s: dict, review: dict, title: str, wrote: str) -> str:
    """정리 보고서 한 장 (사람이 읽는 글). 숫자 표는 코드가 센 것이고, 그 아래 글은 wrote(확인 담당 또는 규칙)가 쓴 것이다."""
    out = [f"# {title}", "", "녹화 영상으로 재현한 시연의 기록입니다. 숫자는 실행 기록에서 코드가 센 것이고, 아래의 글은 그 숫자를 보고 "
           f"{wrote}이(가) 쓴 것입니다. 사람 수가 준 것은 영상이 원래 그런 것이지 요원을 보낸 결과가 아닙니다.", "", "## 기록에서 센 숫자", "", "```", facts_text(s), "```", "",
           "## 무엇이 문제였나", ""] + [f"- {x}" for x in review["problems"]] + ["", "## 무엇이 더 필요했나", ""] + [f"- {x}" for x in review["needs"]] + [
           "", "## 요원 수", "", f"- 판단: **{review['staffing']}**", f"- 까닭: {review['staffing_reason']}", "", "## 지침에 더할 문장 (제안)", "",
           "관리자가 승인한 것만 지침에 들어갑니다.", ""] + [f"- {x}" for x in review["suggestions"]]
    return "\n".join(out) + "\n"


# ---- 결과 보고서 (D-75): 같은 숫자와 글을 보고서 꼴로 보여 준다. 화면(app.py)과 파일이 아래의 같은 줄을 쓴다 ----------------
def headline_numbers(s: dict) -> list[tuple[str, str, str]]:
    """보고서 맨 위의 큰 숫자: (이름, 값, 덧붙이는 말). 모두 run_stats가 기록에서 센 값이다."""
    o = s["orders"]
    wait = "–" if s["first_order_wait_avg"] is None else f"{s['first_order_wait_avg']}초"
    return [
        ("붐빔 사건", f"{s['incidents_opened']}건", f"그 가운데 {s['incidents_closed']}건을 닫음"),
        ("요원에게 보낸 지시", f"{sum(o.values())}건", f"붐빔 대응 {o[CROWD]} · 휴게 교대 {o[RELIEF]} · 인원 채우기 {o[FILL]}"),
        ("도착 뒤 다시 세기", f"{sum(i['rechecks'] for i in s['incidents'])}번", f"사건을 닫고 요원을 돌려보낸 일 {s['returns']}번"),
        ("붐빔을 잡고 첫 지시까지", wait, "사건별 평균 (토론한 시간 포함)"),
        ("규칙이 AI의 판단을 바로잡은 일", f"{s['corrections']}번", f"AI가 답하지 못해 규칙으로 대신한 일 {s['ai_errors']}번"),
        ("관리자에게 넘긴 일", f"{s['no_staff']}번", "규칙을 지키면서 보낼 요원이 없을 때"),
    ]


def incident_rows(s: dict) -> list[dict]:
    """사건별 경과 표의 줄. 모두 run_stats가 기록에서 센 값이다."""
    rows = []
    for i in s["incidents"]:
        result = (f"{i['seconds_to_close']}초 만에 닫음 (평균 {i['before_avg']}명 → {i['after_avg']}명)" if i["closed"]
                  else "기록이 끝날 때까지 닫지 못함")
        rows.append({"zone": f"{i['zone']}구역", "opened": f"{i['opened_clock']} · {i['level']}",
                     "wait": "지시가 나가지 않음" if i["first_order_wait"] is None else f"{i['first_order_wait']}초",
                     "sent": f"{i['sent']}명" + (f" (못 간다고 한 요원 {i['refusals']}명)" if i["refusals"] else ""),
                     "rechecks": f"{i['rechecks']}번", "handed": f"{i['no_staff']}번", "result": result, "closed": i["closed"]})
    return rows


def field_notes(s: dict) -> list[str]:
    """현장에서 들어온 일과 운영 조건. 모두 run_stats가 기록에서 센 값이다."""
    notes = [
        f"현장 사정으로 못 간다고 한 요원 {s['refusals']}명, 다친 사람 보고 {s['injuries']}건 (다친 사람은 관리자에게 따로 알림).",
        f"관리자의 최소 인원 지시 {s['min_staff_changes']}건, 관리자가 취소한 지시 {s['cancels']}건, 최소 인원을 채우지 못한 일 {s['fill_short']}번.",
        f"휴게 시작 {s['breaks_started']}건, 휴게 교대 지시 {s['orders'][RELIEF]}건, 교대 요원을 구하지 못한 일 {s['no_relief']}번.",
        f"예비 요원 {s['reserves_total']}명 가운데 {s['reserves_used']}명이 지시를 받음"
        + (f" ({s['reserves_all_used_clock']}에 마지막 예비 요원까지 지시를 받음)." if s["reserves_all_used_clock"] else "."),
        f"AI 담당들의 토론 {s['debates']}번 가운데 효율 담당과 안전 담당의 선택이 갈린 것 {s['debates_split']}번 "
        f"(계획 담당이 효율 쪽을 따름 {s['planner_followed_efficiency']}번, 안전 쪽을 따름 {s['planner_followed_safety']}번).",
    ]
    if s["staff_script"] or s["scripted_actions"]:
        notes.append(f"요원의 답과 도착 가운데 {s['scripted_actions']}건은 사람이 아니라 정해 둔 대본이 넣음.")
    return notes


REPORT_NOTICE = ("녹화 영상으로 재현한 시연의 기록입니다. 숫자와 표는 실행 기록에서 코드가 센 것이고, ‘확인 담당의 정리’만 그 숫자를 보고 쓴 글입니다. "
                 "사람 수가 준 것은 영상이 원래 그런 것이지 요원을 보낸 결과가 아닙니다.")


def writer_label(wrote: str) -> str:
    """정리 글을 쓴 쪽을 사람이 읽는 말로. AI가 답하지 못해 규칙이 대신 썼으면 그대로 밝힌다."""
    return wrote if wrote.startswith("규칙") else "확인 담당 (AI)"


def result_report_markdown(s: dict, review: dict, made: str, wrote: str, corrected: bool = False) -> str:
    """결과 보고서 한 장을 파일로 남길 글. 숫자와 표는 코드가 센 것이고, '확인 담당의 정리'만 wrote(확인 담당 또는 규칙)가 쓴 것이다."""
    out = ["# 행사 안전 운영 결과 보고서", "", f"- 작성: {made} · 실행 기록 {s['events']}건 · 마지막 기록은 영상 {s['video_seconds']}초",
           f"- 숫자와 표: 실행 기록에서 코드가 셈 / 정리 글: {writer_label(wrote)}", "", f"> {REPORT_NOTICE}", "", "## 1. 한눈에 보기", ""]
    out += [f"- {name}: **{value}** ({note})" for name, value, note in headline_numbers(s)]
    out += ["", "## 2. 사건별 경과", "", "| 구역 | 시작 · 단계 | 첫 지시까지 | 지시받은 요원 | 다시 세기 | 관리자에게 넘김 | 결과 |", "|---|---|---|---|---|---|---|"]
    out += [f"| {r['zone']} | {r['opened']} | {r['wait']} | {r['sent']} | {r['rechecks']} | {r['handed']} | {r['result']} |" for r in incident_rows(s)]
    if not s["incidents"]:
        out.append("| – | 붐빔 사건이 없었음 | – | – | – | – | – |")
    out += ["", "## 3. 현장에서 들어온 일과 운영 조건", ""] + [f"- {x}" for x in field_notes(s)]
    out += ["", "## 4. 확인 담당의 정리", "", "### 무엇이 문제였나", ""] + [f"- {x}" for x in review["problems"]]
    out += ["", "### 무엇이 더 필요했나", ""] + [f"- {x}" for x in review["needs"]]
    out += ["", "### 요원 수", "", f"- 판단: **{review['staffing']}**" + (" (AI의 답이 숫자 기준과 달라 숫자대로 바로잡음)" if corrected else ""),
            f"- 까닭: {review['staffing_reason']}"]
    return "\n".join(out) + "\n"
