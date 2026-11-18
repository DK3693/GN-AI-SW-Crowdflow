# 이 파일이 하는 일: 붐빔 한 건을 처음부터 끝까지 한 번 돌린다 (9단계). 영상 3분을 1초씩 흘려보내면서,
#   관리자와 요원이 할 일(글 쓰기, 도착 누르기)을 정해 둔 대본대로 넣는다. 사람이 중간에 코드를 고치지 않는다.
#   나온 기록(AI가 한 일, 대화방)을 그대로 파일에 남긴다.
# 해당 기능: 핵심 기능 1~4 전체 (E2E)
# 해당 기준: C-02, C-03, C-07, C-10, C-13, C-25, C-28, C-39
# 근거: D-02, D-12 (줄어드는 경우와 줄지 않는 경우를 모두 보여 줌), D-14, 01의 10번 (시연 장면 순서), E-02, E-10
# 출처: AI 코딩 도구로 작성
#
# 실행: PYTHONPATH=src uv run python -m crowdflow.run_e2e            (CrewAI 담당 + Ollama 모델)
#       PYTHONPATH=src uv run python -m crowdflow.run_e2e --rule     (AI 없이 규칙으로만: 흐름 자체를 확인할 때)
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

from crowdflow import chatroom as chat
from crowdflow.crowd_vision import load_yaml
from crowdflow.flow import ARRIVAL, MANAGER_TEXT, STAFF_TEXT, TICK, Operation
from crowdflow.records import read_events
from crowdflow.staffing import load_staff
from crowdflow.timeline import build_timeline
from crowdflow.venue import load_venue

ROOT = Path(__file__).resolve().parents[2]

# 대본: 사람이 할 일을 정해 둔 것. 실제 시연에서는 관리자 화면과 요원 화면에서 사람이 직접 한다.
SCRIPT = {
    "manager_text_at": 20,  # 영상 20초에 관리자가 글을 씀
    "manager_text": "A구역은 2명 유지해",
    "first_staff_reply_after": 5,  # 처음 지시를 받은 요원이 5초 뒤에 글로 답함
    "first_staff_reply": "다친 관람객을 돌보는 중이라 못 갑니다",
    "arrive_after": 15,  # 그다음부터 지시를 받은 요원은 15초 뒤에 '도착'을 누름
}


def run(use_rule: bool, out_dir: Path, seconds: int | None = None, script: dict = SCRIPT, quiet: bool = True, settings: dict | None = None) -> dict:
    settings = settings or load_yaml(ROOT / "config" / "settings.yaml")  # 시험에서는 고정해 둔 설정을 넘긴다
    venue = load_venue(ROOT / settings["venue"])
    timeline = build_timeline(venue, settings["crowd_levels"]["smoothing_seconds"])
    if use_rule:
        from crowdflow.agents import RuleBrain
        brain = RuleBrain()
    else:
        from crowdflow.agents import CrewBrain
        brain = CrewBrain(settings["llm"])

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, room = out_dir / "events.jsonl", chat.ChatRoom(out_dir / "chat.jsonl")
    log_path.unlink(missing_ok=True)
    room.clear()
    op = Operation(settings, timeline, brain, load_staff(ROOT / settings["staffing"]["roster"]), room, log_path)

    def handle(trigger, t, **payload):
        if not quiet:
            return op.handle(trigger, t, **payload)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):  # CrewAI의 진행 표시를 숨김
            return op.handle(trigger, t, **payload)

    total = seconds or int(timeline.n_samples / timeline.sample_rate)
    todo: list[tuple[float, str, dict]] = [(script["manager_text_at"], MANAGER_TEXT, {"text": script["manager_text"]})]
    seen_orders, first_order_done, started = 0, False, time.time()
    for t in range(total):
        for due in sorted((e for e in todo if e[0] <= t), key=lambda e: e[0]):  # 때가 된 일을 넣은 차례대로
            todo.remove(due)
            handle(due[1], float(t), **due[2])
        handle(TICK, float(t))
        orders = [m for m in room.all() if m.kind == chat.ORDER]
        for m in orders[seen_orders:]:  # 새로 나간 지시마다 요원이 할 일을 대본에 넣는다
            if not first_order_done:
                todo.append((t + script["first_staff_reply_after"], STAFF_TEXT, {"staff_id": m.to, "text": script["first_staff_reply"]}))
                first_order_done = True
            else:
                todo.append((t + script["arrive_after"], ARRIVAL, {"staff_id": m.to}))
        seen_orders = len(orders)

    events = read_events(log_path)
    summary = {
        "brain": brain.name, "video_seconds": total, "wall_seconds": round(time.time() - started, 1),
        "events": len(events), "messages": len(room.all()),
        "kinds": [e["kind"] for e in events],
        "closed_incidents": op.state.closed,
        "open_incident": op.state.incident,
        "open_incidents": list(op.state.incidents.values()),
        "final_min_staff": op.state.min_staff,
        "final_staff": [{k: s[k] for k in ("id", "name", "zone", "status", "unavailable_reason")} for s in op.state.staff],
        "script": script,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rule", action="store_true", help="AI 없이 규칙으로만 돌린다")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_dir = ROOT / (args.out or f"outputs/runs/{'rule' if args.rule else 'crew'}")
    summary = run(args.rule, out_dir)
    sys.stdout.reconfigure(encoding="utf-8")
    for e in read_events(out_dir / "events.jsonl"):
        why = f"  ← {e['why']}" if e.get("why") else ""
        print(f"{e['clock']} [{e['who']}] {e['kind']}: {e['what']}{why}"[:400])
    print("\n--- 대화방")
    for m in chat.ChatRoom(out_dir / "chat.jsonl").all():
        print(f"{m.clock} {m.sender} → {m.to} [{m.kind}] {m.text}"[:300])
    print(f"\n{summary['brain']}: 영상 {summary['video_seconds']}초를 {summary['wall_seconds']}초에 돌림, 기록 {summary['events']}건, 대화 {summary['messages']}건")
    print("끝난 사건:", [(c["zone"], c["status"], f"{c['before_avg']}→{c['after_avg']}명", f"다시 확인 {c['rechecks']}회") for c in summary["closed_incidents"]], "| 열린 사건:", summary["open_incident"] and summary["open_incident"]["status"])


if __name__ == "__main__":
    main()
