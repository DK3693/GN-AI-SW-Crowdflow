# 이 파일이 하는 일: 붐빔 한 건을 처음부터 끝까지 처리하는 흐름. 정해진 순서와 상태는 CrewAI Flow가 잡고,
#   판단이 필요한 곳에서만 담당(AI)을 부른다. 사람 세기와 요원 고르기는 도구가 한다.
#     영상(사람 세기 도구) → 붐빔 단계 → 선택지(배치 계산 도구) → 토론과 결정(효율·안전·계획 담당)
#     → 지시(대화방) → 요원의 답 읽기(소통 담당) → 다시 계산 → 도착 → 다시 세기 → 판단(확인 담당)
#   휴게도 여기서 챙긴다: 휴게가 시작되면 기록하고, 붐비는 구역의 요원이 곧 쉬면 바꿔 줄 요원을 보낸다.
# 해당 기능: 핵심 기능 1~4를 잇는 부분
# 해당 기준: C-02, C-03, C-06, C-07, C-09, C-10, C-12, C-22, C-25, C-28, C-36, C-38, C-39, C-44
# 근거: D-02 (핵심 흐름), D-05, D-07, D-08, D-09 (휴게 규칙), D-21, D-41, D-45, D-47, D-49 (담당 구성), D-50 (승인 방식),
#   D-51 (대화방), D-53 (보낼 수 있는 요원이 모자랄 때: 있는 만큼 보내고 알림), D-54 (휴게와 교대를 흐름에 이음),
#   D-57 (여러 구역의 사건을 함께 다룸, 붐비는 구역의 요원은 빼 오지 않음),
#   D-58 (관리자의 취소, 오래 안 풀리면 더 보내기, 풀리면 복귀, 닫은 구역은 잠깐 지켜본 뒤 다시 엶),
#   D-60 (대본이 요원 대신 넣은 답과 도착은 기록에 그렇게 남김),
#   D-62 (관리자가 정한 최소 인원보다 근무 인원이 적으면 채우러 보냄), D-63 (더 보낼 때는 그 까닭을 토론에 알려 줌),
#   D-64 (지침 읽기가 켜져 있으면 토론 전에 운영 지침에서 조항을 찾아 토론에 넘김. 오늘 있었던 일을 토론에 알려 줌. 돌려 본 결과는 E-71),
#   E-35 (CrewAI Flow: @start, @router, @listen, 상태), E-55 (화면에서 돌려 본 결과), 팀 작업 규칙 6번
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai.flow.flow import Flow, listen, router, start  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from crowdflow import chatroom as chat  # noqa: E402
from crowdflow.agents import RuleBrain  # noqa: E402
from crowdflow.levels import level_for, needs_dispatch  # noqa: E402
from crowdflow.records import log_event  # noqa: E402
from crowdflow.staffing import (  # noqa: E402
    MOVING, WORKING, Staff, arrive, clock_to_minutes, dispatch_options, extra_needed, minutes_to_clock,
    minutes_until_break, on_break, on_duty_count, plan_dispatch, start_move,
)

TICK, MANAGER_TEXT, STAFF_TEXT, ARRIVAL, APPROVE, CANCEL = "tick", "manager_text", "staff_text", "arrival", "approve", "cancel"
CROWD, RELIEF, FILL = "붐빔 대응", "휴게 교대", "최소 인원 채우기"  # 요원을 보내는 까닭 세 가지


class OperationState(BaseModel):
    """흐름이 기억하는 것. 일이 생길 때마다(시각이 흐름, 글이 옴, 도착) 흐름을 한 번 돌리고, 이 상태를 이어 쓴다."""

    trigger: str = TICK
    payload: dict[str, Any] = Field(default_factory=dict)
    t: float = 0.0  # 영상의 몇 초인지
    min_staff: dict[str, int] = Field(default_factory=dict)  # 구역별 최소 인원 (관리자의 글로 바뀜)
    staff: list[dict[str, Any]] = Field(default_factory=list)  # 요원 상태
    incidents: dict[str, dict[str, Any]] = Field(default_factory=dict)  # 구역 → 지금 처리 중인 붐빔 사건 (구역마다 하나, 여러 구역이 함께 붐빌 수 있다)
    incident_seq: int = 0  # 사건에 붙이는 번호
    closed_at: dict[str, float] = Field(default_factory=dict)  # 구역 → 그 구역의 사건을 마지막으로 닫은 때 (영상의 몇 초)
    extra: dict[str, list[str]] = Field(default_factory=dict)  # 구역 → 붐빔 대응으로 와 있는(또는 오고 있는) 요원. 다음에 필요한 인원을 셀 때 뺀다
    closed: list[dict[str, Any]] = Field(default_factory=list)  # 끝난 사건 기록
    fills: dict[str, dict[str, Any]] = Field(default_factory=dict)  # 요원 번호 → 관리자가 정한 최소 인원을 채우러 가는 중인 이동 (구역, 출발한 곳)
    injuries: int = 0  # 오늘 들어온 다친 사람 보고의 수 (토론과 지침 찾기에 알려 준다)
    refusals: int = 0  # 오늘 현장 사정으로 못 간다고 한 요원의 수 (토론과 지침 찾기에 알려 준다)
    pending_instruction: dict[str, Any] | None = None  # 관리자 확인을 기다리는 읽은 결과 (승인 방식이 manual일 때)
    on_break: list[str] = Field(default_factory=list)  # 지금 휴게 중인 요원 (휴게 시작과 끝을 한 번씩만 기록하려고)
    breaks_seen: bool = False  # 휴게를 한 번이라도 살펴봤는지 (처음부터 쉬고 있던 요원을 "시작"으로 적지 않으려고)

    @property
    def incident(self) -> dict[str, Any] | None:
        """열려 있는 사건 가운데 가장 먼저 열린 것 (사건이 하나일 때 쓰기 편하게)."""
        return min(self.incidents.values(), key=lambda i: i["id"], default=None)


class Context:
    """흐름이 쓰는 도구와 설정 묶음. 상태가 아니라서 Flow 밖에 둔다."""

    def __init__(self, settings: dict, timeline, brain, room: chat.ChatRoom, log_path: Path, status_fn=None, guideline_fn=None):
        self.settings, self.timeline, self.brain, self.room, self.log_path = settings, timeline, brain, room, log_path
        self.levels, self.staffing, self.op = settings["crowd_levels"], settings["staffing"], settings["operation"]
        self.fallback = RuleBrain()
        self.status_fn = status_fn  # 실시간 분석을 쓸 때: 영상의 몇 초인지를 받아 구역별 상태를 돌려주는 함수
        self.guideline_fn = guideline_fn  # 운영 지침을 읽힐 때: 상황 글을 받아 읽을 조항을 돌려주는 함수. 꺼져 있으면 None을 돌려준다 (D-64)


class CrowdFlow(Flow[OperationState]):
    ctx: Any = Field(default=None, exclude=True)  # 도구와 설정 묶음(Context). 만들 때 넣는다

    # ---- 공통 도우미 -------------------------------------------------------------------------------
    def _now(self) -> float:
        """시연 시계 (자정부터 센 분)."""
        return clock_to_minutes(self.ctx.settings["scenario"]["start_clock"]) + self.state.t / 60

    def _clock(self) -> str:
        s = int(round(self._now() * 60))  # 초로 바꿔 한 번에 반올림한다 (59.6초가 "분은 그대로, 초는 00"으로 찍히던 것을 고침)
        return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"

    def _staff(self) -> list[Staff]:
        return [Staff(**s) for s in self.state.staff]

    def _save_staff(self, staff: list[Staff]) -> None:
        self.state.staff = [asdict(s) for s in staff]

    def _place(self, zone: str) -> str:
        return f"{zone}구역" if zone in self.state.min_staff else zone

    def _log(self, kind: str, who: str, what: str, why: str = "", **data) -> None:
        log_event(kind, who, what, why, path=self.ctx.log_path, clock=self._clock(), t=round(self.state.t, 1), **data)

    def _ask(self, what: str, fn_name: str, *args):
        """담당(AI)을 부른다. 오류가 나면 규칙으로 대신하고, 그 사실을 기록하고 관리자에게 알린다 (C-39, E-08의 'API 오류')."""
        try:
            return getattr(self.ctx.brain, fn_name)(*args), self.ctx.brain.name
        except Exception as e:  # 모델이 꺼져 있음, 시간 초과, 형식에 맞지 않는 답 등
            self._log("AI 오류", "흐름", f"{what}에서 AI가 답하지 못해 규칙으로 대신함", why=f"{type(e).__name__}: {str(e)[:160]}")
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT, f"AI가 답하지 못해 '{what}'을(를) 규칙으로 처리했습니다. 결과를 확인해 주세요.")
            return getattr(self.ctx.fallback, fn_name)(*args), self.ctx.fallback.name

    def _status(self) -> dict[str, dict]:
        if self.ctx.status_fn is not None:  # 화면에서 실시간 분석을 쓸 때는 그쪽이 낸 값을 읽는다
            return self.ctx.status_fn(self.state.t)
        i = min(int(self.state.t * self.ctx.timeline.sample_rate), self.ctx.timeline.n_samples - 1)
        return self.ctx.timeline.status(i, self.ctx.levels)

    # ---- 흐름: 시작 → 갈림길 -----------------------------------------------------------------------
    @start()
    def receive(self):
        return self.state.trigger

    @router(receive)
    def route(self):
        return self.state.trigger

    # ---- 시각이 흐름: 휴게 챙기기 → 사람 세기 → 사건 시작 / 교대 / 다시 세기 ---------------------------------
    @listen(TICK)
    def on_tick(self):
        status = self._status()
        self._breaks()
        for zone, zs in status.items():  # 구역마다 따로 본다. 여러 구역의 사건을 함께 다룬다 (D-57)
            inc = self.state.incidents.get(zone)
            if inc is None:
                # 5초 평균이 다 찬 뒤부터 본다. 방금 닫은 구역은 잠깐 지켜본 뒤에 다시 연다 (금방 오르내리는 값에 휘둘리지 않게, D-58)
                settled = self.state.t >= self.ctx.levels["smoothing_seconds"] and \
                    self.state.t - self.state.closed_at.get(zone, -1e9) >= self.ctx.op["reopen_wait_seconds"]
                if settled and needs_dispatch(level_for(zs["count_avg"], self.ctx.levels), self.ctx.levels):
                    self.state.incident_seq += 1
                    self.state.incidents[zone] = inc = {
                        "id": self.state.incident_seq, "zone": zone, "opened_t": self.state.t, "level": zs["level_name"],
                        "before_avg": round(zs["count_avg"], 1), "sent": [], "arrived": [], "orders": {}, "tasks": {}, "relief": {},
                        "plans": 0, "rechecks": 0, "recheck_at": None, "status": "대응 중",
                    }
                    self._log("붐빔 감지", "사람 세기 도구", f"{zone}구역이 '{zs['level_name']}' 단계가 됨",
                              why=f"구역 안 사람 수의 {self.ctx.levels['smoothing_seconds']}초 평균 {zs['count_avg']:.1f}명 (기준 {self.ctx.levels['levels'][0]['min']}명 이상)",
                              zone=zone, count_avg=round(zs["count_avg"], 1), level=zs["level_name"], incident=inc["id"])
                    self._respond(inc, zs["level_name"])  # 붐빔 대응을 먼저 정하고
                    self._relief(inc)  # 그다음에 휴게 교대를 본다
            else:
                self._relief(inc)
                if inc["recheck_at"] is not None and self.state.t >= inc["recheck_at"]:
                    self._recheck(inc, zs)
        return "ticked"

    def _more_needed(self, inc: dict, level_name: str) -> int:
        """그 단계에서 더 보내야 할 요원 수: 단계별 필요 인원(D-47)에서, 붐빔 대응으로 이미 와 있거나 오고 있는 요원을 뺀다."""
        return max(0, extra_needed(level_name, self.ctx.staffing) - len(self.state.extra.get(inc["zone"], [])))

    def _respond(self, inc: dict, level_name: str) -> None:
        needed = self._more_needed(inc, level_name)
        if needed > 0:
            self._plan(inc, needed)
            return
        here = len(self.state.extra.get(inc["zone"], []))  # 앞선 사건 때 온 요원이 아직 이 구역에 있음
        inc["plans"] += 1
        self._log("계획", "흐름", f"붐빔 대응 요원 {here}명이 이미 {inc['zone']}구역에 있어 더 보내지 않고 지켜봄", why="단계별 필요 인원을 이미 채움 (D-47)",
                  plan_id=f"{inc['id']}-{inc['plans']}", chosen=0, zone=inc["zone"], headline=f"이미 와 있는 요원 {here}명으로 대응")
        inc["recheck_at"] = self.state.t + self.ctx.op["recheck_after_seconds"]
        inc["status"] = "다시 확인 대기"

    def _busy_zones(self, but: str) -> tuple[str, ...]:
        """사건이 열려 있는 다른 구역들. 그 구역의 요원은 빼 오지 않는다."""
        return tuple(z for z in self.state.incidents if z != but)

    # ---- 휴게: 시작과 끝을 기록한다. 휴게는 붐벼도 미루지 않는다 (휴게 규칙 2, D-09) ----------------------------
    def _breaks(self) -> None:
        staff, now, cfg = self._staff(), self._now(), self.ctx.staffing
        resting = [s.id for s in staff if on_break(s, now, cfg)]
        by_id = {s.id: s for s in staff}
        first, self.state.breaks_seen = not self.state.breaks_seen, True
        for s in staff:
            if s.id in resting and s.id not in self.state.on_break:
                end = minutes_to_clock(now + cfg["break_length_minutes"] - (now - s.next_break) % 60)
                if first:  # 시연을 시작할 때 이미 쉬고 있던 요원
                    self._log("휴게 중", "흐름", f"{s.name} 휴게 중 ({end}까지)", why="휴게 중인 요원에게는 지시를 보내지 않음 (휴게 규칙)", staff=s.id, zone=s.zone, until=end)
                    continue
                cover = ""
                for inc in list(self.state.incidents.values()) + self.state.closed[::-1]:  # 이 요원의 교대를 정해 둔 사건이 있으면 함께 알린다
                    if inc and s.id in inc.get("relief", {}):
                        rid = inc["relief"][s.id]
                        cover = (f" 교대 요원 {by_id[rid].name}: {'도착함' if rid in inc['arrived'] else '이동 중'}." if rid
                                 else " 교대할 요원을 구하지 못했습니다.")
                        break
                self._log("휴게 시작", "흐름", f"{s.name} 휴게 시작 ({end}까지).{cover}", why="1시간마다 10분. 정해진 휴게는 붐벼도 미루지 않음 (휴게 규칙)",
                          staff=s.id, zone=s.zone, until=end)
                self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE, f"{s.name}({self._place(s.zone)}) 휴게 시작, {end}까지.{cover} 휴게 중에는 지시를 보내지 않습니다.")
            elif s.id not in resting and s.id in self.state.on_break:
                self._log("휴게 끝", "흐름", f"{s.name} 휴게 끝, 다시 근무", staff=s.id, zone=s.zone)
        self.state.on_break = resting

    def _leaving(self, inc: dict, staff: list[Staff], now: float) -> list[Staff]:
        """사건이 난 구역에서 근무 중인데 곧 쉬는 요원 (아직 교대 요원을 못 정한 사람만)."""
        cfg = self.ctx.staffing
        return [s for s in staff if s.zone == inc["zone"] and s.status == WORKING and not on_break(s, now, cfg)
                and minutes_until_break(s, now) <= cfg["soon_break_minutes"] and not inc["relief"].get(s.id)]

    def _relief(self, inc: dict) -> None:
        """붐비는 구역의 요원이 곧 쉬면 바꿔 줄 요원을 보낸다. 없으면 예비 요원, 그래도 없으면 관리자에게 알린다 (휴게 규칙 4·5).

        AI를 쓰지 않는 규칙 계산이다. 조건(휴게, 최소 인원)은 붐빔 대응과 똑같이 지킨다.
        """
        if inc.get("hold"):  # 관리자가 지시를 취소한 사건에는 AI가 요원을 더 보내지 않는다
            return
        now, cfg = self._now(), self.ctx.staffing
        for leaving in self._leaving(inc, self._staff(), now):
            staff = self._staff()  # 앞에서 보낸 요원의 상태가 바뀌었을 수 있어 다시 읽는다
            when = minutes_to_clock(now + minutes_until_break(leaving, now))
            waiting = tuple(m["staff_id"] for m in inc.get("proposal", []))  # 승인을 기다리는 배치안에 이미 든 요원은 또 고르지 않는다
            plan = plan_dispatch(inc["zone"], 1, staff, now, cfg, dict(self.state.min_staff), purpose=f"{RELIEF} ({leaving.name}이(가) {when}에 휴게)", skip=waiting,
                                 busy_zones=self._busy_zones(inc["zone"]))
            excluded = [{"staff": j.staff_id, "name": j.name, "reason": j.reason} for j in plan.excluded]
            if plan.moves:
                m = plan.moves[0]
                inc["relief"][leaving.id] = m.staff_id
                self._log(RELIEF, "배치 계산 도구", f"{leaving.name}이(가) {when}에 휴게 → {m.name}을(를) 교대 요원으로 보냄",
                          why=m.reason + "".join(f"; {j['name']} 제외: {j['reason']}" for j in excluded if "채워짐" not in j["reason"]),
                          zone=inc["zone"], leaving=leaving.id, staff=m.staff_id, excluded=excluded, incident=inc["id"])
                self._send_or_propose(inc, [dict(asdict(m), purpose=RELIEF, leaving=leaving.id, leaving_name=leaving.name, when=when)],
                                      f"{leaving.name}의 휴게({when}) 교대로 {m.name}을(를) {inc['zone']}구역으로 보냄", crowd=False)
            elif leaving.id not in inc["relief"]:  # 처음 못 구했을 때 한 번만 알리고, 그 뒤로는 조용히 다시 찾아본다
                inc["relief"][leaving.id] = None
                self._log("교대 요원 없음", "흐름", f"{leaving.name}이(가) {when}에 휴게인데 교대할 요원이 없음", why=plan.summary(),
                          zone=inc["zone"], leaving=leaving.id, excluded=excluded, incident=inc["id"])
                self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT,
                                   f"{leaving.name}({inc['zone']}구역)이(가) {when}에 휴게인데 교대할 요원이 없습니다. 직접 정해 주세요. 휴게는 미루지 않습니다.\n{plan.summary()}")

    # ---- 관리자의 글: 소통 담당이 읽음 → 조건을 바꿈 ---------------------------------------------------
    @listen(MANAGER_TEXT)
    def on_manager_text(self):
        text = self.state.payload["text"]
        self._log("관리자 글", chat.MANAGER, f"\"{text}\"")
        read, who = self._ask("관리자 글 읽기", "read_manager_text", text, dict(self.state.min_staff))
        if read.kind != "set_min_staff" or read.zone not in self.state.min_staff or read.min_staff < 0:
            said = "최소 인원 지시가 아니라서 조건을 바꾸지 않았습니다." if read.kind == "other" else "무슨 뜻인지 읽지 못했습니다. 예: \"A구역은 2명 유지해\"처럼 다시 써 주세요."
            self._log("글 읽기", "소통 담당", f"조건으로 바꾸지 않음 ({read.kind})", why=said, by=who, applied=False)
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.QUESTION, said)
            return "not_applied"
        restated = f"{read.zone}구역의 최소 인원을 {self.state.min_staff[read.zone]}명에서 {read.min_staff}명으로 바꿉니다."
        self._log("글 읽기", "소통 담당", restated, why="관리자의 글을 '최소 인원 지시'로 읽음", by=who, zone=read.zone, min_staff=read.min_staff, applied=True)
        if self.ctx.op["approval_mode"] == "manual":  # 관리자가 "예"라고 해야 반영 (D-06)
            self.state.pending_instruction = {"zone": read.zone, "min_staff": read.min_staff, "restated": restated}
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.QUESTION, f"이렇게 읽었습니다: {restated} 맞으면 승인해 주세요.")
            return "waiting_confirmation"
        self._apply_min_staff(read.zone, read.min_staff, restated)
        return "applied"

    def _apply_min_staff(self, zone: str, n: int, restated: str) -> None:
        before = self.state.min_staff[zone]
        self.state.min_staff[zone] = n
        here = on_duty_count(self._staff(), zone, self._now(), self.ctx.staffing)
        self._log("조건 바꿈", "흐름", restated, why="관리자의 지시", zone=zone, min_staff=n, before=before, on_duty=here)
        self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE, f"반영했습니다: {restated}")
        self._fill(zone)

    def _fill(self, zone: str) -> None:
        """관리자가 정한 최소 인원보다 그 구역의 근무 인원이 적으면, 모자란 만큼 채우러 보낸다 (D-62).

        "유지해"는 빼 가지 않는 것만이 아니라 그만큼 있게 하라는 뜻이다. 누구를 보낼지는 배치 계산 도구가 고른다
        (휴게와 다른 구역의 최소 인원을 그대로 지킨다). 보낼 요원이 모자라면 관리자에게 알린다.
        """
        staff, now, need = self._staff(), self._now(), self.state.min_staff[zone]
        coming = sum(1 for s in staff if s.status == MOVING and s.target_zone == zone)  # 이미 이 구역으로 오고 있는 요원
        short = need - on_duty_count(staff, zone, now, self.ctx.staffing) - coming
        if short <= 0:
            return
        plan = plan_dispatch(zone, short, staff, now, self.ctx.staffing, dict(self.state.min_staff), purpose=FILL, busy_zones=self._busy_zones(zone))
        excluded = [{"staff": j.staff_id, "name": j.name, "reason": j.reason} for j in plan.excluded]
        if plan.moves:
            names = ", ".join(m.name for m in plan.moves)
            self._log(FILL, "배치 계산 도구", f"{zone}구역의 근무 인원이 최소 인원({need}명)보다 {short}명 적어 {names}을(를) 보냄",
                      why="; ".join(m.reason for m in plan.moves), zone=zone, short=short, min_staff=need, excluded=excluded,
                      moves=[{"staff": m.staff_id, "name": m.name, "from": m.zone, "travel": m.travel} for m in plan.moves])
        for m in plan.moves:
            s = next(x for x in staff if x.id == m.staff_id)
            origin = s.zone
            start_move(s, zone)
            text = (f"[{s.name}] {origin} → {zone}구역으로 이동해 주세요. 관리자 지시로 {zone}구역의 최소 인원({need}명)을 채웁니다. "
                    "도착하면 '도착'을 눌러 주세요.")
            msg = self.ctx.room.post(self._clock(), chat.CONTROL, s.id, chat.ORDER, text)
            self.state.fills[s.id] = {"zone": zone, "from": origin, "t": self.state.t}
            self._log("지시 보냄", "대화방 도구", f"{s.name}에게 {zone}구역으로 이동하라고 보냄 ({FILL})", why=m.reason, staff=s.id,
                      message_id=msg.id, purpose=FILL, zone=zone, origin=origin)
        self._save_staff(staff)
        if plan.moves:
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE,
                               f"{zone}구역의 최소 인원({need}명)을 채우려고 {', '.join(m.name for m in plan.moves)}을(를) 보냈습니다.")
        if plan.shortfall > 0:  # 조건을 지키면서 보낼 요원이 모자람: 사람에게 알린다
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT,
                               f"{zone}구역의 최소 인원({need}명)에 {plan.shortfall}명이 모자랍니다. 조건을 지키면서 보낼 수 있는 요원이 없습니다. 직접 정해 주세요.\n{plan.summary()}")
            self._log("최소 인원 모자람", "흐름", f"{zone}구역의 최소 인원({need}명)에 {plan.shortfall}명이 모자람", why=plan.summary(),
                      zone=zone, short=plan.shortfall, min_staff=need, excluded=excluded)

    # ---- 승인 (승인 방식이 manual일 때만 쓰임) --------------------------------------------------------
    @listen(APPROVE)
    def on_approve(self):
        if self.state.pending_instruction:
            p, self.state.pending_instruction = self.state.pending_instruction, None
            self._apply_min_staff(p["zone"], p["min_staff"], p["restated"])
        for inc in list(self.state.incidents.values()):
            if inc.get("proposal"):
                self._log("관리자 승인", chat.MANAGER, f"{inc['zone']}구역의 배치안을 승인함", zone=inc["zone"])
                self._dispatch(inc, inc.pop("proposal"))
                inc["status"] = "도착 대기"
        return "approved"

    # ---- 요원의 글: 소통 담당이 읽음 → 못 가면 빼고 다시 계산 -------------------------------------------
    @listen(STAFF_TEXT)
    def on_staff_text(self):
        sid, text = self.state.payload["staff_id"], self.state.payload["text"]
        staff = self._staff()
        s = next(x for x in staff if x.id == sid)
        inc = next((i for i in self.state.incidents.values() if sid in i["tasks"]), None)  # 이 요원에게 지시를 보낸 사건
        order = (inc or {}).get("orders", {}).get(sid, "")
        scripted = bool(self.state.payload.get("scripted"))  # 사람이 아니라 대본이 넣은 글인지 (D-60). 기록에 그대로 남긴다
        self.ctx.room.post(self._clock(), sid, chat.CONTROL, chat.REPLY, text)
        self._log("요원 글", s.name, f"\"{text}\"", why="대본이 요원 대신 넣은 글" if scripted else "", staff=sid, scripted=scripted)
        read, who = self._ask("요원 답 읽기", "read_staff_reply", text, order)
        self._log("글 읽기", "소통 담당", f"{s.name}의 답을 '{ {'yes': '갈 수 있음', 'no': '지금 못 감', 'unclear': '알 수 없음'}[read.can_go] }'으로 읽음",
                  why=read.reason or "이유를 적지 않음", by=who, staff=sid, can_go=read.can_go, injury=read.injury_mentioned)
        if read.injury_mentioned:  # 다친 사람 이야기는 요원을 빼는 것과 별도로 관리자에게 따로 알린다 (D-07)
            self.state.injuries += 1
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT, f"다친 사람이 있다는 보고: {s.name}({self._place(s.zone)}) — \"{text}\"")
            self._log("다친 사람 알림", "소통 담당", f"{self._place(s.zone)}에 다친 사람이 있다고 관리자에게 따로 알림", why=f"{s.name}의 답에 다친 사람 이야기가 있음", zone=s.zone, staff=sid)
        if read.can_go == "unclear":
            self.ctx.room.post(self._clock(), chat.CONTROL, sid, chat.QUESTION, "답을 이해하지 못했습니다. 갈 수 있으면 \"갑니다\", 못 가면 이유를 적어 주세요.")
            self._log("확인 질문", "소통 담당", f"{s.name}에게 다시 물음", why="답의 뜻을 알 수 없음", staff=sid)
        elif read.can_go == "no" and s.status == MOVING:
            task = (inc or {}).get("tasks", {}).pop(sid, None)
            s.unavailable_reason = read.reason or text
            s.status, s.target_zone = WORKING, None  # 이동을 그만두고 제자리에 있음
            self._save_staff(staff)
            self.state.refusals += 1
            fill = self.state.fills.pop(sid, None)  # 최소 인원을 채우러 가던 요원이었는지 (D-62)
            self._log("요원 제외", "흐름", f"{s.name}을(를) 이번 대응에서 뺌", why=f"현장 사정으로 지금 못 감: {s.unavailable_reason}", staff=sid,
                      purpose=(task or {}).get("purpose") or (FILL if fill else None))
            if fill:  # 다른 요원을 다시 찾는다. 없으면 관리자에게 알린다
                self._fill(fill["zone"])
            if task is None:  # 사건이 이미 닫힌 뒤: 이동만 그만둔다
                return "read"
            if sid in inc["sent"]:
                inc["sent"].remove(sid)
            if sid in self.state.extra.get(inc["zone"], []):
                self.state.extra[inc["zone"]].remove(sid)
            inc["orders"].pop(sid, None)
            if inc.get("hold"):
                return "read"
            if task["purpose"] == RELIEF:
                inc["relief"].pop(task["leaving"], None)  # 교대 요원을 다시 찾는다 (없으면 관리자에게 알림)
                self._relief(inc)
            else:
                self._plan(inc, max(self._more_needed(inc, self._status()[inc["zone"]]["level_name"]), 1))
        return "read"

    # ---- 도착: 다시 셀 때를 잡음 -------------------------------------------------------------------
    @listen(ARRIVAL)
    def on_arrival(self):
        sid = self.state.payload["staff_id"]
        staff = self._staff()
        s = next(x for x in staff if x.id == sid)
        if s.status != MOVING:
            return "ignored"
        arrive(s)
        self._save_staff(staff)
        inc = next((i for i in self.state.incidents.values() if sid in i["tasks"]), None)
        task = (inc or {}).get("tasks", {}).get(sid, {})
        fill = self.state.fills.pop(sid, None)  # 관리자가 정한 최소 인원을 채우러 온 요원 (D-62)
        scripted = bool(self.state.payload.get("scripted"))  # 사람이 아니라 대본이 누른 것인지 (D-60)
        self.ctx.room.post(self._clock(), sid, chat.CONTROL, chat.ARRIVED, f"{s.name} {s.zone}구역 도착")
        self._log("도착", s.name, f"{s.zone}구역에 도착했다고 버튼을 누름", why="대본이 요원 대신 누름" if scripted else "", staff=sid, zone=s.zone,
                  purpose=task.get("purpose") or (FILL if fill else None), scripted=scripted)
        if not inc:  # 사건이 이미 닫힌 뒤에 도착
            return "arrived"
        inc["arrived"].append(sid)
        if set(inc["sent"]) <= set(inc["arrived"]) and inc["recheck_at"] is None and not inc.get("proposal"):
            inc["recheck_at"] = self.state.t + self.ctx.op["recheck_after_seconds"]
            inc["status"] = "다시 확인 대기"
            self._log("다시 세기 예약", "흐름", f"{self.ctx.op['recheck_after_seconds']}초 뒤에 {inc['zone']}구역을 다시 셈", why="붐빔 대응으로 보낸 요원이 모두 도착함")
        return "arrived"

    # ---- 관리자의 취소: AI가 보낸 지시를 사람이 멈춘다 (D-58, 대회의 "사용자 확인·안전장치") ------------------------
    @listen(CANCEL)
    def on_cancel(self):
        sid = self.state.payload["staff_id"]
        staff = self._staff()
        s = next(x for x in staff if x.id == sid)
        if s.status != MOVING:
            return "ignored"
        inc = next((i for i in self.state.incidents.values() if sid in i["tasks"]), None)
        s.status, s.target_zone = WORKING, None
        self._save_staff(staff)
        self.ctx.room.post(self._clock(), chat.CONTROL, sid, chat.NOTICE, "관리자가 지시를 취소했습니다. 제자리에서 근무해 주세요.")
        self._log("지시 취소", chat.MANAGER, f"{s.name}에게 보낸 지시를 취소함", why="관리자가 화면에서 취소함. AI의 실행을 사람이 멈출 수 있게 한 안전장치", staff=sid,
                  zone=(inc or {}).get("zone"))
        fill = self.state.fills.pop(sid, None)
        if fill:  # 최소 인원을 채우러 가던 요원: 취소하면 다시 채우지 않는다 (관리자가 멈춘 것이므로). 필요하면 지시를 다시 쓰면 된다
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE,
                               f"{s.name}의 지시를 취소했습니다. {fill['zone']}구역의 최소 인원은 다시 채우지 않습니다. 필요하면 지시를 다시 써 주세요.")
        if inc:
            task = inc["tasks"].pop(sid)
            if sid in inc["sent"]:
                inc["sent"].remove(sid)
            if sid in self.state.extra.get(inc["zone"], []):
                self.state.extra[inc["zone"]].remove(sid)
            inc["orders"].pop(sid, None)
            if task["purpose"] == RELIEF:
                inc["relief"][task["leaving"]] = "관리자가 취소"
            inc["hold"] = True  # 이 사건은 이제 AI가 요원을 더 보내지 않고 지켜보기만 한다
            inc["status"] = "관리자가 맡음"
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE,
                               f"{s.name}의 지시를 취소했습니다. {inc['zone']}구역 사건에는 AI가 요원을 더 보내지 않고 지켜보기만 합니다.")
            if set(inc["sent"]) <= set(inc["arrived"]) and inc["recheck_at"] is None:
                inc["recheck_at"] = self.state.t + self.ctx.op["recheck_after_seconds"]
        return "cancelled"

    # ---- 선택지 → 토론 → 결정 → 지시 ----------------------------------------------------------------
    def _plan(self, inc: dict, needed: int) -> None:
        staff, now = self._staff(), self._now()
        options = dispatch_options(inc["zone"], needed, staff, now, self.ctx.staffing, dict(self.state.min_staff), busy_zones=self._busy_zones(inc["zone"]))
        inc["plans"] = inc.get("plans", 0) + 1
        pid = f"{inc['id']}-{inc['plans']}"  # 같은 계산에서 나온 기록(선택지, 토론, 결정, 지시)을 화면에서 묶는 번호
        for o in options:
            self._log("배치 계산", "배치 계산 도구", o.describe() if o.plan.solved else o.plan.summary(),
                      why="; ".join(f"{j.name} 제외: {j.reason}" for j in o.plan.excluded) or "뺀 요원 없음", option=o.number, title=o.title,
                      plan_id=pid, solved=o.plan.solved, needed=needed, zone=inc["zone"], facts=o.facts,
                      moves=[{"staff": m.staff_id, "name": m.name, "from": m.zone, "travel": m.travel} for m in o.plan.moves],
                      excluded=[{"staff": j.staff_id, "name": j.name, "reason": j.reason} for j in o.plan.excluded])
        short = options[0].plan
        if not short.solved and not short.moves:  # 조건을 지키면서 보낼 요원이 한 명도 없음: 사람에게 넘긴다
            inc["status"] = "관리자에게 넘김"
            if not inc.get("no_staff_alerted"):  # 같은 사건에서 같은 알림을 되풀이하지 않는다
                inc["no_staff_alerted"] = True
                self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT,
                                   f"{inc['zone']}구역에 보낼 수 있는 요원이 없습니다. 직접 정해 주세요.\n{short.summary()}")
                self._log("관리자에게 넘김", "흐름", "조건을 지키면서 보낼 수 있는 요원이 없음", why=short.summary(), plan_id=pid, zone=inc["zone"])
            if inc["recheck_at"] is None and set(inc["sent"]) <= set(inc["arrived"]):  # 보낼 요원이 없어도 계속 지켜본다 (줄어들면 닫는다)
                inc["recheck_at"] = self.state.t + self.ctx.op["recheck_after_seconds"]
            return
        if not short.solved:  # 필요한 인원보다 적게만 보낼 수 있음: 있는 만큼 보내고, 모자란 인원은 관리자에게 알린다 (D-53)
            names = ", ".join(m.name for m in short.moves)
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT,
                               f"{inc['zone']}구역에 {needed}명이 필요하지만 조건을 지키면서 보낼 수 있는 요원은 {len(short.moves)}명뿐입니다. "
                               f"{names}을(를) 먼저 보냅니다. 모자란 {short.shortfall}명은 직접 정해 주세요.\n{short.summary()}")
            self._log("계획", "흐름", f"필요한 {needed}명 중 {len(short.moves)}명만 보낼 수 있어, {names}을(를) 먼저 보내고 모자란 {short.shortfall}명은 관리자에게 알림",
                      why="선택지를 만들 수 없어 토론 없이 정함. 조건(휴게, 최소 인원)은 그대로 지킴", plan_id=pid, chosen=1, zone=inc["zone"],
                      headline=f"보낼 수 있는 {len(short.moves)}명만 먼저 보냄")
            self._send_or_propose(inc, [dict(asdict(m), plan_id=pid) for m in short.moves], f"{names}을(를) {inc['zone']}구역으로 보냄 (필요한 {needed}명 중 {len(short.moves)}명)")
            return
        status = self._status()[inc["zone"]]
        situation = (f"{inc['zone']}구역의 사람 수 {self.ctx.levels['smoothing_seconds']}초 평균이 {status['count_avg']:.1f}명으로 '{status['level_name']}' 단계. "
                     f"기준에 따라 요원 {needed}명이 더 필요함. 지금 시각 {self._clock()[:5]}. "
                     f"구역별 최소 인원: {', '.join(f'{z} {n}명' for z, n in self.state.min_staff.items())}.")
        if inc["rechecks"] > 0:  # 이미 대응하고도 풀리지 않아 더 보내는 경우: 토론이 그 사실을 알고 따지게 한다 (D-63)
            situation += (f" 참고: 이 구역에는 붐빔 대응 요원 {len(self.state.extra.get(inc['zone'], []))}명이 이미 와 있고, "
                          f"{inc['rechecks']}번 다시 세었지만 아직 풀리지 않았음.")
        # 오늘 있었던 일을 알려 준다: 다친 사람 보고, 못 간다고 한 요원, 남은 예비 요원 (예비 요원을 남길지 따지고, 지침의 조건과 맞춰 볼 수 있게. E-71)
        today = self._today(staff)
        if today:
            situation += f" 참고: {today}"
        others = [f"{z}구역({i['level']})" for z, i in self.state.incidents.items() if z != inc["zone"]]
        if others:  # 다른 구역도 붐비면 알려 준다. 그 구역의 요원은 선택지에 들어 있지 않다
            situation += f" 참고: {', '.join(others)}도 붐벼 대응 중이라, 그 구역의 요원은 빼 오지 않음."
        leaving = self._leaving(inc, staff, now)
        if leaving:  # 곧 쉬는 요원이 있으면 토론에 알려 준다. 교대 요원은 이 결정 뒤에 계산 도구가 따로 정한다
            situation += f" 참고: 이 구역의 {', '.join(s.name for s in leaving)}이(가) 곧 휴게라, 이 결정 뒤에 교대 요원 {len(leaving)}명을 따로 보내야 함."
        if len(options) == 1:
            chosen, pick = options[0], 1
            self._log("계획", "계획 담당", "선택지가 하나뿐이라 토론 없이 선택지 1로 정함", why=chosen.describe(), plan_id=pid, chosen=1, zone=inc["zone"],
                      headline="보낼 수 있는 요원이 이 선택지뿐임")
        else:
            options_text = "\n".join(o.describe() for o in options)
            found = None
            # 지침을 찾을 때의 물음은 "지금 정할 것"을 한 문장으로 쓴다. 상황과 선택지를 통째로 넣으면 엉뚱한 조항이 위로 왔다 (E-71)
            query = (f"붐비는 구역('{status['level_name']}' 단계)에 요원 {needed}명을 더 보내야 한다. 누구를 먼저 보낼지, 몇 명을 보낼지 정해야 한다. "
                     f"선택지: {' / '.join(o.title for o in options)}."
                     + (" 이미 요원을 보냈는데도 풀리지 않아 더 보내는 것이다." if inc["rechecks"] > 0 else "")
                     + (f" {today}" if today else ""))
            if self.ctx.guideline_fn:  # 운영 지침에서 지금 상황에 맞는 조항을 찾아 온다 (D-64). 찾지 못해도 토론은 그대로 한다
                try:
                    found = self.ctx.guideline_fn(query)
                except Exception as e:
                    self._log("지침 찾기 오류", "흐름", "운영 지침을 찾지 못해 지침 없이 토론함", why=f"{type(e).__name__}: {str(e)[:160]}", plan_id=pid)
            clauses = (found or {}).get("clauses") or None
            if clauses:
                self._log("지침 찾기", "지침 찾기 도구", f"운영 지침 {found['total']}개 조항 가운데 {len(clauses)}개를 읽음 ({found['mode']})",
                          why="; ".join(f"[{c['id']}] {c['text']}" for c in clauses), plan_id=pid, mode=found["mode"], total=found["total"], clauses=clauses)
            (decision, transcript), who = self._ask("배치 토론", "debate", situation, options_text, clauses)
            for line in transcript:
                said = "해당하는 조항이 없다고 함" if line["who"] == "지침 담당" and not line["option"] else f"선택지 {line['option']}을(를) 고름"
                self._log("토론", line["who"], said, why=line["says"], by=who, plan_id=pid,
                          option=line["option"], point=line.get("point", ""), clauses=line.get("clauses"), model=line.get("model"))
            pick = decision.chosen_option if 1 <= decision.chosen_option <= len(options) else 1
            if pick != decision.chosen_option:  # 없는 번호를 고르면 기본 계산을 쓴다 (C-44)
                self._log("AI 답 바로잡음", "흐름", "계획 담당이 없는 선택지 번호를 골라 선택지 1을 씀", why=f"고른 번호 {decision.chosen_option}", plan_id=pid)
            chosen = options[pick - 1]
            self._log("계획", "계획 담당", f"선택지 {pick} ({chosen.title})로 정함", why=decision.reason, steps=decision.steps, by=who, plan_id=pid,
                      chosen=pick, zone=inc["zone"], headline=decision.headline,
                      situation=situation, options_text=options_text, query=query)  # 같은 상황을 지침 없이/읽고 다시 물어 견줄 수 있게 남긴다 (D-64)
        self._send_or_propose(inc, [dict(asdict(m), plan_id=pid) for m in chosen.plan.moves], chosen.describe())

    def _today(self, staff: list[Staff]) -> str:
        """오늘 지금까지 있었던 일 한 줄 (기록에 있는 숫자만). 없으면 빈 글."""
        parts = []
        if self.state.injuries:
            parts.append(f"오늘 다친 사람 보고가 {self.state.injuries}건 있었음.")
        if self.state.refusals:
            parts.append(f"오늘 현장 사정으로 못 간다고 한 요원이 {self.state.refusals}명 있었음.")
        reserves = [s for s in staff if s.reserve]
        out = [s for s in reserves if s.status == MOVING or s.unavailable_reason or s.zone != self.ctx.staffing["standby_zone"]]
        if out:
            parts.append(f"예비 요원 {len(reserves)}명 가운데 {len(out)}명은 출동했거나 지금 못 가고, 대기 중인 예비 요원은 {len(reserves) - len(out)}명임.")
        return " ".join(parts)

    def _send_or_propose(self, inc: dict, moves: list[dict], described: str, crowd: bool = True) -> None:
        """승인 방식에 따라 바로 지시를 보내거나(auto), 관리자의 승인을 기다린다(manual) (D-50)."""
        if self.ctx.op["approval_mode"] == "manual":
            inc.setdefault("proposal", []).extend(moves)
            inc["status"] = "승인 대기"
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.QUESTION, f"배치안: {described} 승인해 주세요.")
        else:
            self._dispatch(inc, moves)
            if crowd:
                inc["status"] = "도착 대기"

    def _dispatch(self, inc: dict, moves: list[dict]) -> None:
        staff = self._staff()
        sent: dict[str, list[str]] = {}
        for m in moves:
            s = next(x for x in staff if x.id == m["staff_id"])
            purpose = m.get("purpose", CROWD)
            start_move(s, inc["zone"])
            why = f"{m['leaving_name']}의 휴게({m['when']}) 교대입니다." if purpose == RELIEF else f"{inc['zone']}구역이 붐빕니다."
            text = f"[{s.name}] {s.zone} → {inc['zone']}구역으로 이동해 주세요. {why} 도착하면 '도착'을 눌러 주세요."
            msg = self.ctx.room.post(self._clock(), chat.CONTROL, s.id, chat.ORDER, text)
            for members in self.state.extra.values():  # 다른 구역에 붐빔 대응으로 가 있던 요원이면 거기서 뺀다
                if s.id in members:
                    members.remove(s.id)
            if purpose == CROWD:
                inc["sent"].append(s.id)
                self.state.extra.setdefault(inc["zone"], []).append(s.id)
            inc["orders"][s.id] = text
            inc["tasks"][s.id] = {"purpose": purpose, "leaving": m.get("leaving"), "from": s.zone, "t": self.state.t}
            sent.setdefault(purpose, []).append(s.name)
            self._log("지시 보냄", "대화방 도구", f"{s.name}에게 {inc['zone']}구역으로 이동하라고 보냄 ({purpose})", why=m["reason"], staff=s.id,
                      message_id=msg.id, purpose=purpose, plan_id=m.get("plan_id"), zone=inc["zone"], origin=s.zone, leaving=m.get("leaving"))
        self._save_staff(staff)
        if self.ctx.op["approval_mode"] != "manual":
            for purpose, names in sent.items():
                self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE,
                                   f"{inc['zone']}구역에 {', '.join(names)}을(를) 보냈습니다 ({purpose}). (관리자 승인 없이 자동으로 보냈습니다.)")

    # ---- 다시 세기 → 확인 담당의 판단 ----------------------------------------------------------------
    def _recheck(self, inc: dict, status: dict) -> None:
        inc["rechecks"] += 1
        inc["recheck_at"] = None
        op = self.ctx.op
        level = level_for(status["count_avg"], self.ctx.levels)
        below = not needs_dispatch(level, self.ctx.levels)
        here = len(self.state.extra.get(inc["zone"], []))
        more = self._more_needed(inc, level.name)
        long_alert = not below and more == 0 and inc["rechecks"] % op["more_after_rechecks"] == 0 and here < op["max_extra_staff"]
        if long_alert:
            more = 1  # 오래 풀리지 않으면 한 명을 더 보낼 수 있다 (D-58). 보낼지는 확인 담당이 정한다
        if inc.get("hold"):
            more = 0  # 관리자가 지시를 취소한 사건: AI는 더 보내지 않는다
        facts = (f"{inc['zone']}구역에 요원이 도착한 뒤 다시 셈. 보내기 전 평균 {inc['before_avg']}명({inc['level']}) → 지금 평균 {status['count_avg']:.1f}명({level.name}). "
                 f"요원을 보내는 기준은 {self.ctx.levels['dispatch_from']}({self.ctx.levels['levels'][0]['min']}명 이상). "
                 f"지금 단계는 {'기준 아래' if below else '기준 이상'}. 붐빔 대응으로 와 있는 인원 {here}명, 더 필요한 인원: {more}명. "
                 f"다시 확인한 횟수 {inc['rechecks']}회.")
        if long_alert and more:
            facts += f" 이 구역은 {inc['rechecks']}번 다시 세어도 풀리지 않았음. 오래 풀리지 않으면 요원을 1명 더 보낼 수 있음(붐빔 대응 요원은 최대 {op['max_extra_staff']}명)."
        if inc.get("hold"):
            facts += " 관리자가 이 사건의 지시를 취소했으므로 요원을 더 보내지 않음."
        self._log("다시 세기", "사람 세기 도구", f"{inc['zone']}구역 평균 {status['count_avg']:.1f}명, '{level.name}' 단계", why=f"보내기 전 {inc['before_avg']}명({inc['level']})",
                  zone=inc["zone"], count_avg=round(status["count_avg"], 1), level=level.name, before=inc["before_avg"], incident=inc["id"])
        verdict, who = self._ask("결과 확인", "judge", facts)
        decision = "keep_watching" if verdict.decision == "notify_manager" else verdict.decision  # 관리자에게 알리는 것은 아래에서 횟수로 정한다
        # 숫자로 정해지는 부분은 규칙이 이긴다: 기준 아래면 닫고, 기준 이상이면 닫지 않는다. 더 보낼 인원이 없으면 보내지 않는다 (C-44)
        expected = "close" if below else ("send_more" if more > 0 else "keep_watching")
        if (decision == "close") != below or (decision == "send_more" and more == 0):
            self._log("AI 답 바로잡음", "흐름", f"확인 담당의 판단({decision})이 숫자와 맞지 않아 규칙({expected})을 따름", why=facts)
            decision = expected
        self._log("결과 확인", "확인 담당", {"close": "사건을 닫음", "send_more": "요원을 더 보냄", "keep_watching": "더 지켜봄"}[decision],
                  why=verdict.reason, by=who, decision=decision, zone=inc["zone"], count_avg=round(status["count_avg"], 1), level=level.name,
                  incident=inc["id"], long_alert=bool(long_alert and more))
        if decision == "close":
            self._release(inc, level.name, status["count_avg"])
            return
        if inc["rechecks"] == op["max_rechecks"]:  # 여러 번 확인해도 풀리지 않음: 관리자에게 한 번 알리고, 계속 지켜본다
            self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.ALERT,
                               f"{inc['zone']}구역이 {inc['rechecks']}번 다시 확인해도 풀리지 않았습니다. AI는 계속 지켜봅니다. 필요하면 직접 정해 주세요.")
            self._log("관리자에게 알림", "흐름", f"{inc['zone']}구역이 {inc['rechecks']}번 확인해도 풀리지 않아 관리자에게 알림", why="계속 지켜보면서 사람의 판단을 구함", zone=inc["zone"])
        if decision == "send_more":
            sent_before = len(inc["sent"])
            self._plan(inc, more)
            if len(inc["sent"]) == sent_before and not inc.get("proposal"):  # 더 보낼 요원이 없음: 알림은 나갔고, 계속 지켜본다
                inc["recheck_at"] = self.state.t + op["recheck_after_seconds"]
                inc["status"] = "다시 확인 대기"
        else:
            inc["recheck_at"] = self.state.t + op["recheck_after_seconds"]

    def _release(self, inc: dict, level_name: str, count_avg: float) -> None:
        """사건을 닫고, 붐빔 대응으로 와 있던 요원을 원래 자리로 돌려보낸다 (사람이 줄면 인원도 줄인다, D-58)."""
        zone = inc["zone"]
        inc["status"] = "종료"
        inc["closed_t"], inc["after_avg"] = self.state.t, round(count_avg, 1)
        self.state.closed.append(inc)
        del self.state.incidents[zone]
        self.state.closed_at[zone] = self.state.t
        staff = self._staff()
        back, stopped = [], []
        for sid in self.state.extra.get(zone, []):
            s = next(x for x in staff if x.id == sid)
            home = inc["tasks"].get(sid, {}).get("from")
            if s.status == MOVING and s.target_zone == zone:  # 아직 오는 중이던 요원: 이동을 그만두게 한다
                s.status, s.target_zone = WORKING, None
                stopped.append(s.name)
                self.ctx.room.post(self._clock(), chat.CONTROL, sid, chat.NOTICE, f"{zone}구역이 풀려 이동을 취소합니다. 제자리에서 근무해 주세요.")
            elif s.zone == zone and s.status == WORKING and home and home != zone:
                start_move(s, home)
                back.append(s.name)
                self.ctx.room.post(self._clock(), chat.CONTROL, sid, chat.ORDER,
                                   f"[{s.name}] {zone}구역이 풀렸습니다. {self._place(home)}(으)로 복귀해 주세요. 도착하면 '도착'을 눌러 주세요.")
        self.state.extra[zone] = []
        self._save_staff(staff)
        said = (f" {', '.join(back)} 복귀 지시." if back else "") + (f" {', '.join(stopped)} 이동 취소." if stopped else "")
        if back or stopped:
            self._log("복귀 지시", "흐름", f"{zone}구역이 풀려 붐빔 대응 인원을 줄임:{said}", why=f"평균 {count_avg:.1f}명, '{level_name}' 단계로 내려감", zone=zone)
        self.ctx.room.post(self._clock(), chat.CONTROL, chat.MANAGER, chat.NOTICE,
                           f"{zone}구역이 '{level_name}' 단계로 내려가 사건을 닫았습니다.{said} (녹화 영상이라, 요원을 보내서 줄었다고 볼 수는 없습니다.)")


class Operation:
    """한 번의 운영: 상태를 들고 있다가, 일이 생길 때마다 흐름(CrowdFlow)을 한 번 돌린다."""

    def __init__(self, settings: dict, timeline, brain, staff: list[Staff], room: chat.ChatRoom, log_path: Path, status_fn=None, guideline_fn=None):
        self.ctx = Context(settings, timeline, brain, room, log_path, status_fn, guideline_fn)
        self.state = OperationState(min_staff=dict(settings["staffing"]["min_staff_per_zone"]), staff=[asdict(s) for s in staff])

    def handle(self, trigger: str, t: float, **payload) -> OperationState:
        flow = CrowdFlow(ctx=self.ctx)
        inputs = self.state.model_dump()
        inputs.update(trigger=trigger, t=t, payload=payload)
        inputs.pop("id", None)
        flow.kickoff(inputs=inputs)
        self.state = flow.state
        return self.state
