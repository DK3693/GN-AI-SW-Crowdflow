# 이 파일이 하는 일: 일어난 일(붐빔, 계산 결과, 승인, 지시, 답, 도착, 다시 센 결과)을 한 줄에 한 건씩 파일에 쌓는다.
#   관리자 화면의 "AI가 한 일 기록"과 사건 기록이 이 파일에서 나온다.
# 해당 기능: 핵심 기능 4 (실행과 재확인)의 기록 부분, 기억/상태
# 해당 기준: C-38 (기억/상태), C-24, 팀 작업 규칙 7번 (기록은 처음부터 남긴다)
# 근거: D-13 (AI가 한 일 기록), D-47의 10번 (한 줄에 한 건씩 쌓는 글 파일), E-07 (실행 기록을 보여 줄 것)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "outputs" / "logs" / "events.jsonl"


def log_event(kind: str, who: str, what: str, why: str = "", path: Path = DEFAULT_LOG, **data) -> dict:
    """기록 한 건을 쌓는다.

    kind: 무슨 일인지 (예: "붐빔 감지", "배치 계산", "관리자 승인")
    who: 누가 했는지 (예: "사람 세기 도구", "배치 계산 도구", "관리자", "소통 담당")
    what: 무엇을 했는지 한 줄로
    why: 왜 그렇게 정했는지 (근거가 된 값이나 조건)
    data: 그 밖에 남길 값 (구역, 요원, 사람 수 등)
    """
    event = {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind, "who": who, "what": what, "why": why, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def read_events(path: Path = DEFAULT_LOG) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
