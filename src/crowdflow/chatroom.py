# 이 파일이 하는 일: 상황실과 요원들이 함께 쓰는 대화방 하나를 파일로 구현한다. 지시 보내기, 요원의 글 답, "도착" 누르기,
#   관리자에게 가는 알림이 모두 이 대화방에 한 줄씩 쌓인다. 요원용 화면과 관리자 화면이 같은 파일을 본다.
# 해당 기능: 핵심 기능 4 (실행과 재확인)의 메시지 부분
# 해당 기준: C-23, C-30, C-37
# 근거: D-51 (메신저는 우리 서버의 대화방 하나로 함. 텔레그램 D-18은 쓰지 않음), D-10 (도착은 버튼으로 확인),
#   D-09의 6번 (휴게 중인 요원에게는 지시를 보내지 않음 — 보내는 쪽인 flow.py가 지킨다)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONTROL = "상황실"  # 지시를 보내는 쪽
MANAGER = "관리자"  # 알림을 받는 사람

ORDER, REPLY, ARRIVED, ALERT, QUESTION, NOTICE = "지시", "답", "도착", "알림", "확인 질문", "안내"


@dataclass
class Message:
    id: int
    clock: str  # 시연 시계의 시각 (예: "20:01:05")
    sender: str  # "상황실" 또는 요원 번호
    to: str  # 요원 번호, "관리자" 또는 "모두"
    kind: str  # 지시 / 답 / 도착 / 알림 / 확인 질문 / 안내
    text: str
    order_id: int | None = None  # 어느 지시에 대한 것인지


class ChatRoom:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Message]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [Message(**json.loads(line)) for line in f if line.strip()]

    def post(self, clock: str, sender: str, to: str, kind: str, text: str, order_id: int | None = None) -> Message:
        msg = Message(len(self.all()) + 1, clock, sender, to, kind, text, order_id)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")
        return msg

    def since(self, last_id: int) -> list[Message]:
        return [m for m in self.all() if m.id > last_id]

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
