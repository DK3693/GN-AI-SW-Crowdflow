# 이 파일이 하는 일: 붐비는 구역에 누구를 보낼지 계산한다. 조건을 지키는 요원만 후보로 삼고, 고른 이유와
#   뺀 이유를 함께 낸다. 조건을 다 지키는 답이 없으면 "답 없음"과 그 이유를 낸다. AI를 쓰지 않는 규칙 계산이다.
# 해당 기능: 핵심 기능 3 (요원 배치 계산)
# 해당 기준: C-23, C-35, C-37, C-44
# 근거: D-08 (요원을 고르는 일은 계산 도구가 함), D-09 (휴게 규칙), D-10 (위치는 구역 단위),
#   D-47 (필요한 인원, 곧 휴게, 이동 시간, 최소 인원, 고르는 순서 — 사용자가 정한 시연 설정), E-17, E-18,
#   D-49 (조건을 지키는 선택지를 만들어 담당에게 넘김), D-57 (붐비는 구역의 요원은 빼 오지 않음),
#   D-61 (예비 요원이 여럿일 때의 선택지)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

WORKING, MOVING = "근무", "이동 중"


def clock_to_minutes(clock: str) -> float:
    """ "20:05" → 자정부터 센 분."""
    h, m = clock.split(":")
    return int(h) * 60 + int(m)


def minutes_to_clock(minutes: float) -> str:
    m = int(minutes) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


@dataclass
class Staff:
    id: str
    name: str
    zone: str  # 지금 있는 구역 ("A", "B", "C" 또는 예비 요원이 기다리는 곳)
    reserve: bool
    next_break: float  # 다음 휴게가 시작되는 시각 (자정부터 센 분). 휴게는 1시간마다 되풀이된다 (D-09)
    status: str = WORKING  # 근무 / 이동 중. 휴게인지는 시각으로 계산한다
    target_zone: str | None = None  # 이동 중이면 가는 곳
    unavailable_reason: str | None = None  # 현장 사정으로 지금 못 가는 이유 (요원의 답에서 온다, D-05)


def load_staff(path: str | Path) -> list[Staff]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [
            Staff(id=r["id"], name=r["name"], zone=r["zone"], reserve=r["reserve"].strip().lower() in ("yes", "true", "1"),
                  next_break=clock_to_minutes(r["next_break"]), status=r["status"])
            for r in csv.DictReader(f)
        ]


def minutes_until_break(s: Staff, now: float) -> float:
    """다음 휴게가 시작될 때까지 남은 분. 휴게 중이면 0. 휴게는 1시간마다 같은 분에 시작한다 (D-09)."""
    since = (now - s.next_break) % 60  # 가장 최근 휴게가 시작된 뒤로 지난 분
    return 60 - since


def on_break(s: Staff, now: float, cfg: dict) -> bool:
    return (now - s.next_break) % 60 < cfg["break_length_minutes"]


def travel_seconds(a: str, b: str, cfg: dict) -> float | None:
    """구역 사이 이동 시간(초). 설정에 없으면 None (정해지지 않음)."""
    if a == b:
        return 0
    t = cfg["travel_seconds"]
    return t.get(f"{a}-{b}", t.get(f"{b}-{a}"))


def on_duty_count(staff: list[Staff], zone: str, now: float, cfg: dict) -> int:
    """그 구역에서 지금 근무 중인 요원 수 (휴게 중이거나 다른 곳으로 이동 중인 요원은 세지 않는다)."""
    return sum(1 for s in staff if s.zone == zone and s.status == WORKING and not on_break(s, now, cfg))


@dataclass
class Judgement:
    """요원 한 명에 대한 판단: 골랐는지, 뺐는지, 그 이유."""

    staff_id: str
    name: str
    zone: str
    chosen: bool
    reason: str
    travel: float | None = None


@dataclass
class DispatchPlan:
    zone: str  # 요원을 보낼 구역
    purpose: str  # "붐빔 대응" 또는 "휴게 교대"
    needed: int  # 더 필요한 인원
    moves: list[Judgement] = field(default_factory=list)  # 고른 요원
    excluded: list[Judgement] = field(default_factory=list)  # 뺀 요원과 이유
    shortfall: int = 0  # 조건을 지키면서는 채우지 못한 인원 (0이 아니면 "답 없음")

    @property
    def solved(self) -> bool:
        return self.shortfall == 0

    def summary(self) -> str:
        lines = [f"[{self.purpose}] {self.zone}구역에 {self.needed}명 필요"]
        lines += [f"  선택: {j.name} ({j.zone} → {self.zone}) — {j.reason}" for j in self.moves]
        lines += [f"  제외: {j.name} — {j.reason}" for j in self.excluded]
        if not self.solved:
            lines.append(f"  답 없음: 조건을 지키면서 보낼 수 있는 요원이 {self.shortfall}명 모자람. 관리자에게 알림")
        return "\n".join(lines)


def extra_needed(level_name: str, cfg: dict) -> int:
    """그 단계에서 구역에 더 필요한 요원 수 (D-47). 표에 없는 단계(평상)는 0."""
    return int(cfg["extra_staff_per_level"].get(level_name, 0))


def plan_dispatch(zone: str, needed: int, staff: list[Staff], now: float, cfg: dict,
                  min_staff: dict[str, int], purpose: str = "붐빔 대응", skip: tuple[str, ...] = (),
                  busy_zones: tuple[str, ...] = ()) -> DispatchPlan:
    """zone으로 needed명을 보낼 계획을 세운다.

    뺄 요원 (하나라도 해당하면 뺀다, D-47의 8번):
      현장 사정으로 못 감 / 이미 이동 중 / 휴게 중 / 곧 휴게 / 빼면 그 구역의 최소 인원이 깨짐
    남은 요원을 고르는 순서: 예비 요원은 맨 나중 → 이동 시간이 짧은 순 → 번호 순
    """
    plan = DispatchPlan(zone=zone, purpose=purpose, needed=needed)
    candidates = []
    for s in staff:
        if s.id in skip or (s.zone == zone and s.status == WORKING) or s.target_zone == zone:
            continue  # 이미 그 구역에 있거나 가고 있는 요원
        travel = travel_seconds(s.zone, zone, cfg)
        reason = None
        if s.zone in busy_zones and s.status == WORKING:
            reason = f"{s.zone}구역도 붐벼 대응 중이라 빼 오지 않음"
        elif s.unavailable_reason:
            reason = f"현장 사정으로 지금 못 감 ({s.unavailable_reason})"
        elif s.status == MOVING:
            reason = f"이미 {s.target_zone}구역으로 이동 중 (한 사람에게 두 가지 일을 주지 않음)"
        elif on_break(s, now, cfg):
            reason = "휴게 중 (휴게 중인 요원에게는 지시를 보내지 않음)"
        elif minutes_until_break(s, now) <= cfg["soon_break_minutes"]:
            reason = f"곧 휴게 ({minutes_until_break(s, now):.0f}분 뒤 {minutes_to_clock(now + minutes_until_break(s, now))})"
        elif s.zone in min_staff and on_duty_count(staff, s.zone, now, cfg) - 1 < min_staff[s.zone]:
            reason = f"빼면 {s.zone}구역의 최소 인원({min_staff[s.zone]}명)이 깨짐"
        if reason:
            plan.excluded.append(Judgement(s.id, s.name, s.zone, False, reason, travel))
        else:
            candidates.append((s, travel))

    # 고르는 순서. 이동 시간이 정해지지 않은 요원(예비)은 같은 무리 안에서 뒤로 보낸다.
    candidates.sort(key=lambda c: (c[0].reserve, c[1] is None, c[1] or 0, c[0].id))
    taken_from: dict[str, int] = {}
    for s, travel in candidates:
        if len(plan.moves) == needed:
            plan.excluded.append(Judgement(s.id, s.name, s.zone, False, "조건은 지키지만 필요한 인원이 이미 채워짐", travel))
            continue
        # 같은 구역에서 여러 명을 빼면 최소 인원이 깨질 수 있으므로, 앞에서 고른 인원까지 넣어 다시 본다
        if s.zone in min_staff and on_duty_count(staff, s.zone, now, cfg) - taken_from.get(s.zone, 0) - 1 < min_staff[s.zone]:
            plan.excluded.append(Judgement(s.id, s.name, s.zone, False, f"빼면 {s.zone}구역의 최소 인원({min_staff[s.zone]}명)이 깨짐", travel))
            continue
        taken_from[s.zone] = taken_from.get(s.zone, 0) + 1
        how = "이동 시간이 정해지지 않음" if travel is None else f"이동 {travel:g}초"
        why = "다른 후보가 없어 예비 요원을 씀" if s.reserve else "조건을 모두 지키는 요원 중 가장 가까움"
        plan.moves.append(Judgement(s.id, s.name, s.zone, True, f"{why} ({how})", travel))
    plan.shortfall = needed - len(plan.moves)
    return plan


def plan_relief(staff: list[Staff], crowded_zones: list[str], now: float, cfg: dict, min_staff: dict[str, int]) -> list[DispatchPlan]:
    """붐비는 구역에 있는 요원이 곧 쉴 때, 바꿔 줄 요원을 찾는다 (휴게 규칙 4·5번, D-09).

    휴게는 미루지 않는다. 바꿔 줄 요원이 없으면 계획의 shortfall이 남고, 관리자에게 알려야 한다.
    """
    plans = []
    for zone in crowded_zones:
        leaving = [s for s in staff if s.zone == zone and s.status == WORKING and not on_break(s, now, cfg)
                   and minutes_until_break(s, now) <= cfg["soon_break_minutes"]]
        if leaving:
            names = ", ".join(s.name for s in leaving)
            plans.append(plan_dispatch(zone, len(leaving), staff, now, cfg, min_staff, purpose=f"휴게 교대 ({names}이(가) 곧 휴게)"))
    return plans


def start_move(s: Staff, zone: str) -> None:
    """지시를 받아 이동을 시작한다."""
    s.status, s.target_zone = MOVING, zone


def arrive(s: Staff) -> None:
    """도착 확인 (요원이 "도착" 버튼을 누름, D-10)."""
    s.zone, s.status, s.target_zone = s.target_zone, WORKING, None


@dataclass
class DispatchOption:
    """조건을 모두 지키는 선택지 하나. 담당(AI)들은 이 선택지들 가운데서만 고른다 (D-49)."""

    number: int
    title: str
    plan: DispatchPlan
    facts: dict  # 선택지를 견줄 때 쓰는 계산값 (지어낸 평가가 아니라 계산 도구가 낸 숫자)

    def describe(self) -> str:
        who = ", ".join(f"{m.name}({m.zone}구역)" if m.zone in self.facts["min_staff"] else f"{m.name}({m.zone})" for m in self.plan.moves)
        f = self.facts
        travel = "정해지지 않음" if f["longest_travel_seconds"] is None else f"{f['longest_travel_seconds']:g}초"
        left = ", ".join(f"{z}구역 {n}명(최소 {f['min_staff'][z]}명)" for z, n in f["on_duty_after"].items())
        return (f"선택지 {self.number} — {self.title}: {who}을(를) {self.plan.zone}구역으로 보냄. "
                f"가장 오래 걸리는 이동 {travel}. 보낸 뒤 남는 근무 인원: {left}. "
                f"예비 요원 {'씀' if f['uses_reserve'] else '남겨 둠'}. "
                f"보내는 요원의 다음 휴게까지 가장 짧게 남은 시간 {f['soonest_break_minutes']:.0f}분.")


def _facts(plan: DispatchPlan, staff: list[Staff], now: float, cfg: dict, min_staff: dict[str, int]) -> dict:
    by_id = {s.id: s for s in staff}
    travels = [m.travel for m in plan.moves]
    return {
        "people": len(plan.moves),
        "longest_travel_seconds": None if any(t is None for t in travels) else max(travels, default=0),
        "uses_reserve": any(by_id[m.staff_id].reserve for m in plan.moves),
        "on_duty_after": {z: on_duty_count(staff, z, now, cfg) - sum(m.zone == z for m in plan.moves) for z in min_staff if z != plan.zone},
        "min_staff": dict(min_staff),
        "soonest_break_minutes": min((minutes_until_break(by_id[m.staff_id], now) for m in plan.moves), default=0.0),
    }


def dispatch_options(zone: str, needed: int, staff: list[Staff], now: float, cfg: dict, min_staff: dict[str, int],
                     busy_zones: tuple[str, ...] = ()) -> list[DispatchOption]:
    """zone에 needed명을 보내는 선택지들을 만든다. 모든 선택지는 반드시 지킬 조건(휴게, 최소 인원 등)을 지킨다.

    1. 가까운 근무 요원을 보냄 (기본 계산, D-47의 고르는 순서 그대로)
    2. 예비 요원을 먼저 보냄 (다른 구역의 인원을 그대로 둠)
    3. 한 명을 더 보냄 (여유를 둠)
    같은 요원 구성이 되거나, 조건을 지키면서 만들 수 없는 선택지는 내지 않는다.
    """
    options: list[DispatchOption] = []
    seen: set[tuple[str, ...]] = set()

    def add(title: str, plan: DispatchPlan):
        key = tuple(sorted(m.staff_id for m in plan.moves))
        if plan.solved and plan.moves and key not in seen:
            seen.add(key)
            options.append(DispatchOption(len(options) + 1, title, plan, _facts(plan, staff, now, cfg, min_staff)))

    base = plan_dispatch(zone, needed, staff, now, cfg, min_staff, busy_zones=busy_zones)
    reserve_ids = {s.id for s in staff if s.reserve}
    only_reserves = bool(base.moves) and all(m.staff_id in reserve_ids for m in base.moves)  # 보낼 수 있는 근무 요원이 없어 예비 요원만 고른 경우
    add("예비 요원을 보냄" if only_reserves else "가까운 근무 요원을 보냄", base)
    regulars = tuple(s.id for s in staff if not s.reserve)
    # 예비 요원만으로 채울 수 있을 때만 선택지가 된다 (모자라면 add가 받지 않는다). 예비 요원이 여럿이면 2명 이상이 필요할 때도 나온다 (D-61)
    add("예비 요원을 먼저 보냄", plan_dispatch(zone, needed, staff, now, cfg, min_staff, skip=regulars, busy_zones=busy_zones))
    add("여유를 두고 한 명을 더 보냄", plan_dispatch(zone, needed + 1, staff, now, cfg, min_staff, busy_zones=busy_zones))
    if not options:  # 조건을 지키는 선택지가 하나도 없음: "답 없음"인 기본 계산을 그대로 돌려준다
        options.append(DispatchOption(1, "보낼 수 있는 요원이 모자람", base, _facts(base, staff, now, cfg, min_staff)))
    return options
