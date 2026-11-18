# 이 파일이 하는 일: 화면(관리자 화면, 요원 대화방)에서 쓰는 "실시간 시연 한 번". 영상 시계를 실제 시간대로 흘리면서
#   1초마다 흐름(flow.py)을 돌리고, 화면에서 온 일(관리자의 글, 요원의 글, 도착)을 온 차례대로 흐름에 넣는다.
#   run_e2e.py는 사람이 할 일을 대본으로 넣지만, 여기서는 사람이 화면에서 직접 한다.
#   흐름은 뒤에서 도는 일꾼(스레드) 하나만 만진다. 그래서 여러 화면이 동시에 눌러도 차례가 섞이지 않는다.
#   사람 수는 실시간 분석 일꾼(vision_live.py, GPU)이 켜져 있으면 거기서 받고, 없으면 미리 분석해 둔 결과를 읽는다.
# 해당 기능: 핵심 기능 1~4를 화면에 잇는 부분
# 해당 기준: C-16, C-23, C-28, C-31, C-38, C-39
# 근거: D-13 (관리자 화면), D-51 (우리 서버의 대화방 하나), D-52 (Streamlit), D-53 (화면 구성),
#   D-55 (실시간 분석을 화면에 이음. 사용자 지시 2026-10-04: CCTV 영상과 YOLO 결과를 실시간으로, GPU를 최대한 활용),
#   D-12 (녹화 영상으로 재현), E-53 (Streamlit은 화면마다 따로 돌기 때문에 함께 쓰는 것은 한 곳에 둬야 함),
#   E-54 (쉬던 모델을 다시 올리는 데 69초가 걸림 → 미리 올려 둠. Ollama 공식 문서의 preload와 keep_alive),
#   D-60 (요원 역할을 대본이 대신하는 스위치. 화면에서 돌려 본 결과는 E-67), E-06 (대본임을 기록과 화면에 표시),
#   D-64 (하루 정리 쓰기, 승인한 문장을 지침에 더하기, 지침 읽기 스위치, "어려운 날" 대본, 같은 상황을 지침 없이/읽고 묻기. 결과는 E-71)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import atexit
import contextlib
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from crowdflow import chatroom as chat
from crowdflow.crowd_vision import load_yaml
from crowdflow.flow import ARRIVAL, CROWD, FILL, STAFF_TEXT, TICK, Operation
from crowdflow.guideline import GuidelineIndex, add_learned, clear_learned, load_clauses
from crowdflow.levels import level_for
from crowdflow.records import log_event, read_events
from crowdflow.review import facts_text, report_markdown, run_stats
from crowdflow.staffing import MOVING, Staff, clock_to_minutes, load_staff, minutes_to_clock, minutes_until_break, on_break, travel_seconds
from crowdflow.timeline import build_timeline
from crowdflow.venue import load_venue

ROOT = Path(__file__).resolve().parents[2]


class VisionLink:
    """실시간 분석 일꾼(vision_live.py)을 따로 띄우고, 거기서 구역별 사람 수와 사람 점을 받아 온다.

    일꾼은 다른 프로세스다. GPU로 사람을 찾는 일이 화면 서버를 느리게 하지 않게 하려는 것이다.
    """

    def __init__(self, cfg: dict, host: str = "127.0.0.1"):
        self.port = int(cfg["port"])
        self.base = f"http://127.0.0.1:{self.port}"
        self._cache: tuple[float, dict | None] = (0.0, None)
        self.proc = None
        if self._call("/health") is None:  # 이미 떠 있는 일꾼이 없을 때만 새로 띄운다
            args = [sys.executable, "-m", "crowdflow.vision_live", "--port", str(self.port), "--host", host, "--device", str(cfg["device"]),
                    "--half" if cfg.get("half") else "--no-half", "--width", str(cfg["stream_width"])]
            log = open(ROOT / "outputs" / "vision_live.log", "ab")
            self.proc = subprocess.Popen(args, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            atexit.register(self.close)

    def _call(self, path: str, body: dict | None = None, timeout: float = 0.6) -> dict | None:
        try:
            req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:  # 아직 켜지는 중, 꺼짐 등
            return None

    def state(self) -> dict | None:
        """일꾼이 낸 가장 새 결과. 잠깐(0.25초) 동안은 받아 둔 것을 다시 쓴다."""
        at, value = self._cache
        if time.monotonic() - at > 0.25:
            value = self._call("/state")
            self._cache = (time.monotonic(), value)
        return value

    def clock(self, t: float, running: bool) -> None:
        self._call("/clock", {"t": round(t, 3), "running": running})

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class LiveOperation:
    def __init__(self, use_rule: bool = False, out_dir: Path | None = None, speed: float = 1.0, vision: bool = False, vision_host: str = "127.0.0.1"):
        """speed: 영상 시계가 실제 시간보다 몇 배 빨리 가는지. 화면에서는 1(실제 시간), 시험에서만 빠르게 돌린다.
        vision: 실시간 분석 일꾼을 쓸지. 시험에서는 쓰지 않고 미리 분석해 둔 결과를 읽는다."""
        self.settings = load_yaml(ROOT / "config" / "settings.yaml")
        self.venue = load_venue(ROOT / self.settings["venue"])
        self.timeline = build_timeline(self.venue, self.settings["crowd_levels"]["smoothing_seconds"])
        self.total = self.timeline.n_samples / self.timeline.sample_rate  # 영상 전체 길이(초)
        if use_rule:
            from crowdflow.agents import RuleBrain
            self.brain = RuleBrain()
        else:
            from crowdflow.agents import CrewBrain
            self.brain = CrewBrain(self.settings["llm"])
        self.speed = speed
        self.out_dir = out_dir or ROOT / "outputs" / "live"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "events.jsonl"
        self.room = chat.ChatRoom(self.out_dir / "chat.jsonl")
        self.vision = VisionLink(self.settings["vision_live"], vision_host) if vision else None
        self._arrived_at: dict[str, float] = {}  # 요원이 방금 도착한 때 (약도에서 도착을 눈에 띄게 그리려고)
        self._was_moving: set[str] = set()
        # 요원 역할을 대본이 대신할지 (D-60). 화면의 스위치가 이 값을 바꾸고, 뒤의 일꾼이 다음 차례에 알아챈다
        self.staff_script: bool = bool(self.settings["scenario"]["staff_script"]["enabled"])
        # "어려운 날" 대본 (D-64): 다친 사람을 돌보느라 못 간다고 답하는 요원이 더 많은 날. 프로그램은 그대로이고 현장 사정(대본)만 어렵다
        self.hard_day: bool = False
        # 운영 지침과 하루 정리 (D-64). 시험에서는(out_dir을 줄 때) 배운 지침과 정리를 그 폴더 안에만 둔다
        self.guide_cfg = dict(self.settings["guideline"])
        if out_dir is not None:
            self.guide_cfg["learned"] = str(Path(out_dir) / "learned.jsonl")
        self.guideline_on: bool = bool(self.guide_cfg["enabled"])  # 토론에서 운영 지침을 읽힐지. 화면에서 켜고 끈다
        self.index = GuidelineIndex(self.guide_cfg, self.settings["llm"]["base_url"])
        self.reports_dir = self.out_dir / "reports"
        self.reviews: list[dict] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.reports_dir.glob("review_*.json"))]  # 지난 정리들 (다시 켜도 남는다)
        # 같은 상황을 지침 없이/읽고 물어본 결과들 (파일로도 남긴다)
        self.compares: list[dict] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.reports_dir.glob("compare_*.json"))]
        self.reviewed_gen = 0  # 정리를 쓴 실행의 번호. 같은 실행을 두 번 정리해 날짜가 늘어나는 일을 막는다
        self._gen = 0  # 처음부터 다시 할 때마다 1씩 오른다. 예전 일꾼은 이 값이 달라진 것을 보고 멈춘다
        self._thread: threading.Thread | None = None
        # AI 담당을 한 번에 한 곳에서만 부르게 한다. 흐름은 뒤의 일꾼이, 정리 쓰기와 견주기는 화면 쪽이 부르는데,
        # 같은 담당을 동시에 부르면 CrewAI가 오류를 낸다 (설치한 crewai의 실행기에 잠금이 있음, E-72)
        self._ai_lock = threading.Lock()
        # AI 모델이 준비됐는지: None = 올리는 중, True = 준비됨, False = AI 서버에 닿지 않음 (그때는 규칙으로 대신한다)
        self.model_ready: bool | None = True
        if not use_rule and self.settings["llm"]["model"].startswith("ollama"):
            self.model_ready = None
            threading.Thread(target=self._keep_model_warm, daemon=True, name="crowdflow-warm").start()
        self.reset()

    # ---- 시계 -----------------------------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._started_at is not None

    @property
    def t(self) -> float:
        """지금이 영상의 몇 초인지. 시작 전에는 0, 영상이 끝나면 끝 시각에 머문다."""
        if self._started_at is None:
            return 0.0
        return min((time.monotonic() - self._started_at) * self.speed, self.total)

    @property
    def finished(self) -> bool:
        return self.started and self.t >= self.total

    def now_minutes(self, t: float | None = None) -> float:
        return clock_to_minutes(self.settings["scenario"]["start_clock"]) + (self.t if t is None else t) / 60

    def clock(self, t: float | None = None) -> str:
        """시연 시계 (settings.yaml의 scenario.start_clock에서 시작)."""
        s = int(round(self.now_minutes(t) * 60))  # 초로 바꿔 한 번에 반올림한다 (59.6초가 "분은 그대로, 초는 00"으로 찍히던 것을 고침)
        return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"

    # ---- 화면이 부르는 것 --------------------------------------------------------------------------
    def reset(self) -> None:
        """처음부터 다시: 기록과 대화방을 비우고, 요원과 조건을 처음 상태로 돌린다."""
        self._gen += 1
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.settings["llm"]["timeout_seconds"] + 30)  # AI가 답하는 중이면 끝날 때까지 기다린다
        self._inbox: queue.Queue = queue.Queue()
        self.waiting: list[tuple[str, dict]] = []  # 화면에서 왔지만 아직 흐름이 처리하지 않은 일
        self.busy: tuple[str, float] | None = None  # (지금 처리 중인 일, 시작한 때)
        self._started_at: float | None = None
        self._arrived_at.clear()
        self._was_moving.clear()
        self._script_on = False  # 뒤의 일꾼이 지금 대본을 돌리고 있는지 (스위치와 다르면 켜거나 끄는 일을 한다)
        self._script_seen = 0  # 대화방의 지시 가운데 대본이 이미 본 수
        self._script_refusals = 0  # 대본이 지금까지 "못 갑니다"라고 답하게 한 요원 수 (정해 둔 수까지만 한다)
        self._script_due: dict[str, tuple[float, str, dict]] = {}  # 요원 번호 → (할 때, 할 일, 값)
        self.log_path.unlink(missing_ok=True)
        self.room.clear()
        self.op = Operation(self.settings, self.timeline, self.brain, load_staff(ROOT / self.settings["staffing"]["roster"]),
                            self.room, self.log_path, status_fn=self.status_at, guideline_fn=self._guideline)
        if self.vision:
            self.vision.clock(0.0, False)
        self._thread = threading.Thread(target=self._loop, args=(self._gen,), daemon=True, name="crowdflow-live")
        self._thread.start()

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = time.monotonic()
            if self.vision:
                self.vision.clock(0.0, True)

    def close(self) -> None:
        """뒤에서 도는 일꾼을 멈춘다 (시험이 끝났을 때)."""
        self._gen += 1

    def submit(self, trigger: str, **payload) -> None:
        """화면에서 온 일을 줄에 세운다. 뒤의 일꾼이 온 차례대로 흐름에 넣는다."""
        self.waiting.append((trigger, payload))
        self._inbox.put((trigger, payload))

    # ---- 사람 수: 실시간 분석이 켜져 있으면 거기서, 아니면 미리 분석해 둔 결과에서 ---------------------------------
    def _live(self) -> dict | None:
        st = self.vision.state() if self.vision else None
        return st["zones"] if st and set(st.get("zones", {})) >= set(self.venue.zones) else None

    def vision_info(self) -> dict:
        """화면에 보여 줄 실시간 분석의 상태."""
        st = self.vision.state() if self.vision else None
        return {"on": self.vision is not None, "live": self._live() is not None, "rate": (st or {}).get("rate"),
                "port": self.vision.port if self.vision else None}

    def status_at(self, t: float) -> dict[str, dict]:
        """영상 t초의 구역별 상태. 흐름(flow.py)과 화면이 같은 값을 본다."""
        live = self._live()
        if live is None:
            i = min(int(t * self.timeline.sample_rate), self.timeline.n_samples - 1)
            return self.timeline.status(i, self.settings["crowd_levels"])
        out = {}
        for k in self.venue.zones:
            lv = level_for(float(live[k]["count_avg"]), self.settings["crowd_levels"])
            out[k] = {"count": int(live[k]["count"]), "count_avg": float(live[k]["count_avg"]), "level_index": lv.index, "level_name": lv.name,
                      "area_m2": None, "density": None}
        return out

    def view(self) -> tuple[dict, dict]:
        """지금 시각의 (구역별 사람 점, 구역별 상태). 약도를 그릴 때 쓴다."""
        live = self._live()
        if live is not None:
            return {k: np.array(live[k]["points"], dtype=float).reshape(-1, 2) for k in self.venue.zones}, self.status_at(self.t)
        i = min(int(self.t * self.timeline.sample_rate), self.timeline.n_samples - 1)
        return {k: tr.at(i) for k, tr in self.timeline.tracks.items()}, self.timeline.status(i, self.settings["crowd_levels"])

    # ---- 요원: 화면에 그릴 수 있게 풀어 쓴 상태 ---------------------------------------------------------
    def staff_view(self, refresh: float = 1.0) -> list[dict]:
        cfg, now, t = self.settings["staffing"], self.now_minutes(), self.t
        state = self.op.state
        tasks: dict[str, dict] = {}
        for inc in state.closed + list(state.incidents.values()):
            tasks.update(inc.get("tasks", {}))
        rows = []
        for d in state.staff:
            s = Staff(**d)
            moving, resting = s.status == MOVING, on_break(s, now, cfg)
            if moving:
                self._was_moving.add(s.id)
            elif s.id in self._was_moving:  # 방금 도착함 (또는 이동을 그만둠)
                self._was_moving.discard(s.id)
                self._arrived_at[s.id] = t
            left = minutes_until_break(s, now)
            row = {"id": s.id, "name": s.name, "zone": s.zone, "reserve": s.reserve, "state": "이동" if moving else "휴게" if resting else "근무",
                   "soon": not resting and left <= cfg["soon_break_minutes"], "blocked": bool(s.unavailable_reason), "reason": s.unavailable_reason or "",
                   "moving": moving, "target": s.target_zone, "fresh": t - self._arrived_at.get(s.id, -99.0) < 1.6,
                   "next_break": minutes_to_clock(now + left), "break_left": left,
                   "rest_left": cfg["break_length_minutes"] - (now - s.next_break) % 60 if resting else None, "purpose": ""}
            if moving:  # 실제 위치는 모른다(구역 단위, D-10). 도착은 요원이 단추를 눌러야 확정된다
                task = tasks.get(s.id, {})
                row["purpose"] = FILL if s.id in state.fills else "복귀" if not task or task.get("from") == s.target_zone else task.get("purpose", "")
            rows.append(row)
        return rows

    # ---- 운영 지침과 하루 정리 (D-64) ---------------------------------------------------------------
    def clauses(self) -> list:
        """지금의 운영 지침 조항들 (기본 지침 + 관리자가 승인해 더한 문장)."""
        return load_clauses(ROOT, self.guide_cfg)

    def _guideline(self, query: str) -> dict | None:
        """토론 전에 흐름이 부른다. 지침 읽기가 꺼져 있으면 None, 켜져 있으면 상황에 맞춰 읽을 조항."""
        return self.index.find(query, self.clauses()) if self.guideline_on else None

    def make_review(self) -> dict:
        """지금까지의 기록으로 하루 정리를 만든다. 숫자는 코드가 기록에서 세고(review.py), 글은 확인 담당이 그 숫자만 보고 쓴다."""
        from crowdflow.agents import RuleBrain

        events = read_events(self.log_path)
        stats = run_stats(events, [d["id"] for d in self.op.state.staff if d["reserve"]])
        facts, wrote = facts_text(stats), self.brain.name
        # 토론으로 정한 결정들: "처음부터 다시"를 눌러 기록이 지워진 뒤에도 같은 상황을 다시 물어볼 수 있게 정리에 함께 남긴다
        plans = [{"clock": e.get("clock"), "zone": e.get("zone"), "chosen": e.get("chosen"), "what": e["what"], "headline": e.get("headline", ""),
                  "situation": e["situation"], "options_text": e["options_text"], "query": e["query"]} for e in events if e["kind"] == "계획" and e.get("situation")]
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self._ai_lock:
                    review = self.brain.review(facts, stats)
        except Exception:  # AI가 답하지 못함: 정해 둔 문장으로 대신 쓴다
            review, wrote = RuleBrain().review(facts, stats), "규칙 (AI가 답하지 못함)"
        # 요원 수의 판단은 숫자로 정해진다. AI의 답이 다르면 숫자를 따른다 (C-44)
        short = stats["no_staff"] + stats["no_relief"] + stats["fill_short"]
        expected = "모자람" if short else "넉넉함" if stats["reserves_used"] < stats["reserves_total"] else "적당함"
        corrected = review.staffing != expected
        review.staffing = expected
        day = len(self.reviews) + 1
        item = {"day": day, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "wrote": wrote, "stats": stats, "review": review.model_dump(),
                "staffing_corrected": corrected, "guideline_on": self.guideline_on, "clauses": len(self.clauses()), "staff_script": self.staff_script,
                "hard_day": self.hard_day, "script_refusals": self.script_refusals() if self.staff_script else 0, "brain": self.brain.name, "plans": plans}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (self.reports_dir / f"review_{stamp}.json").write_text(json.dumps(item, ensure_ascii=False, indent=1), encoding="utf-8")
        (self.reports_dir / f"review_{stamp}.md").write_text(report_markdown(stats, item["review"], f"{day}일차 정리 ({item['made']})", wrote), encoding="utf-8")
        self.reviews.append(item)
        self.reviewed_gen = self._gen
        log_event("하루 정리", "규칙" if wrote.startswith("규칙") else "확인 담당", f"{day}일차 정리를 씀: 요원 수는 '{expected}'",
                  why=review.staffing_reason, path=self.log_path, clock=self.clock(), t=round(self.t, 1), day=day, by=wrote, corrected=corrected)
        return item

    def approve(self, texts: list[str], source: str) -> int:
        """관리자가 승인한(또는 직접 쓴) 문장을 운영 지침에 더한다. 더한 개수를 돌려준다."""
        n = add_learned(ROOT, self.guide_cfg, texts, source)
        if n:
            log_event("지침 더함", "관리자", f"운영 지침에 문장 {n}개를 더함", why=" / ".join(t.strip() for t in texts if t.strip()),
                      path=self.log_path, clock=self.clock(), t=round(self.t, 1), source=source)
        return n

    def forget(self) -> None:
        """승인해 더한 지침과 지난 정리를 모두 지운다 (기본 지침은 그대로). 시연을 처음부터 다시 보여 줄 때 쓴다."""
        clear_learned(ROOT, self.guide_cfg)
        for p in list(self.reports_dir.glob("review_*")) + list(self.reports_dir.glob("compare_*")):
            p.unlink()
        self.reviews.clear()
        self.compares.clear()
        self.reviewed_gen = 0

    def compare(self, situation: str, options_text: str, query: str) -> dict:
        """같은 상황과 같은 선택지를, 운영 지침 없이 한 번, 읽히고 한 번 토론시켜 견준다. 달라지는 것은 지침을 읽었는지뿐이다."""
        found = self.index.find(query, self.clauses())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self._ai_lock:
                plain = self.brain.debate(situation, options_text, None)
                guided = self.brain.debate(situation, options_text, found["clauses"])
        item = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "situation": situation, "options_text": options_text, "query": query, "found": found,
                "without": {"chosen": plain[0].chosen_option, "headline": plain[0].headline, "reason": plain[0].reason, "lines": plain[1]},
                "with": {"chosen": guided[0].chosen_option, "headline": guided[0].headline, "reason": guided[0].reason, "lines": guided[1]}}
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / f"compare_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(item, ensure_ascii=False, indent=1), encoding="utf-8")
        self.compares.append(item)
        return item

    # ---- AI 모델을 미리 올려 두기 --------------------------------------------------------------------
    def _keep_model_warm(self) -> None:
        """Ollama에 빈 요청을 보내 모델을 올려 두고(preload), 내려가지 않게 일정한 간격으로 되풀이한다 (E-54)."""
        cfg = self.settings["llm"]
        names = [cfg["model"].split("/", 1)[1]]
        op = cfg.get("opinion") or {}
        if op.get("enabled"):  # 효율·안전 의견을 내는 가벼운 모델도 함께 올려 둔다 (D-65)
            names.append(op["model"].split("/", 1)[1])
        while True:
            try:
                for name in names:
                    body = json.dumps({"model": name, "keep_alive": cfg["keep_alive"]}).encode()
                    req = urllib.request.Request(cfg["base_url"] + "/api/generate", data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=cfg["timeout_seconds"]) as r:
                        r.read()
                self.model_ready = True
            except Exception:  # Ollama가 꺼져 있음 등. 흐름은 규칙으로 대신 돌고(flow.py의 _ask), 다음 차례에 다시 해 본다
                self.model_ready = False
            time.sleep(cfg["warm_every_seconds"])

    # ---- 뒤에서 도는 일꾼 --------------------------------------------------------------------------
    def _loop(self, gen: int) -> None:
        last_tick = -1
        while self._gen == gen:
            did = False
            try:
                trigger, payload = self._inbox.get_nowait()
            except queue.Empty:
                pass
            else:
                self._handle(trigger, payload)
                if (trigger, payload) in self.waiting:
                    self.waiting.remove((trigger, payload))
                did = True
            t = self.t
            if self.started and int(t) > last_tick:  # 영상 1초마다 한 번. AI가 답하느라 늦어지면 지나간 초는 건너뛴다
                last_tick = int(t)
                if self.vision:
                    self.vision.clock(t, not self.finished)  # 실시간 분석 일꾼의 시계를 이쪽 시계에 맞춘다
                self._handle(TICK, {})
                self._play_staff_script()
                did = True
            if not did:
                time.sleep(0.05)

    # ---- 요원 역할을 대본이 대신하기 (D-60) ---------------------------------------------------------------
    def script_refusals(self) -> int:
        """대본에서 "못 갑니다"라고 답하는 요원 수. 평소 대본과 "어려운 날" 대본이 다르다."""
        cfg = self.settings["scenario"]["staff_script"]
        return int(cfg["hard_refusals"] if self.hard_day else cfg["refusals"])

    def _play_staff_script(self) -> None:
        """사람이 요원 화면에서 하던 일을 정해 둔 대로 넣는다. 영상 1초마다 한 번 불린다.

        붐빔 대응 지시를 처음 받은 요원 몇 명(평소 대본 1명, "어려운 날" 대본은 더 많이)은 "못 갑니다"라고 답하고,
        그 밖의 요원은 이동 시간이 지나면 "도착"을 누른다.
        AI가 하는 일이 아니다. 대본이 넣은 것은 기록에 "대본"이라고 남고, 켜 있는 동안 화면에 표시된다 (E-06: 꾸미지 않기).
        """
        t = self.t
        if self.staff_script != self._script_on:  # 스위치가 바뀜
            self._script_on = self.staff_script
            self._script_due.clear()
            self._script_seen = sum(1 for m in self.room.all() if m.kind == chat.ORDER)  # 켠 때부터 나가는 지시에 적용한다
            if self._script_on:
                for d in self.op.state.staff:  # 이미 이동 중인 요원은 지금부터 이동 시간이 지나면 도착시킨다
                    if d["status"] == MOVING:
                        self._script_due[d["id"]] = (t + (travel_seconds(d["zone"], d["target_zone"], self.settings["staffing"]) or 0), ARRIVAL,
                                                      {"staff_id": d["id"], "scripted": True})
            log_event("요원 역할 대본", "관리자", "요원 역할을 대본이 대신함: " + ("켬" if self._script_on else "끔"),
                      why=f"붐빔 대응 지시를 처음 받은 요원 {self.script_refusals()}명은 못 간다고 답하고, 그 밖의 요원은 이동 시간이 지나면 도착을 누름. "
                          "사람이 요원 화면에서 하는 일을 대신한 것이고 AI의 판단이 아님",
                      path=self.log_path, clock=self.clock(t), t=round(t, 1), on=self._script_on, refusals=self.script_refusals(), hard_day=self.hard_day)
        if not self._script_on:
            return
        cfg = self.settings["scenario"]["staff_script"]
        staff = {d["id"]: d for d in self.op.state.staff}
        orders = [m for m in self.room.all() if m.kind == chat.ORDER]
        for m in orders[self._script_seen:]:  # 새로 나간 지시마다 그 요원이 할 일을 잡는다. 같은 요원의 예전 일은 새 일로 바꾼다
            d = staff.get(m.to)
            if d is None or d["status"] != MOVING:
                continue
            crowd = any(i["tasks"].get(m.to, {}).get("purpose") == CROWD for i in self.op.state.incidents.values())
            if crowd and self._script_refusals < self.script_refusals():
                self._script_refusals += 1
                self._script_due[m.to] = (t + cfg["first_reply_after_seconds"], STAFF_TEXT, {"staff_id": m.to, "text": cfg["first_reply"], "scripted": True})
            else:
                self._script_due[m.to] = (t + (travel_seconds(d["zone"], d["target_zone"], self.settings["staffing"]) or 0), ARRIVAL,
                                          {"staff_id": m.to, "scripted": True})
        self._script_seen = len(orders)
        for sid, (due, trigger, payload) in list(self._script_due.items()):
            if t >= due:
                del self._script_due[sid]
                self.submit(trigger, **payload)

    def _handle(self, trigger: str, payload: dict) -> None:
        t = self.t
        self.busy = (trigger, time.monotonic())
        try:
            # CrewAI의 진행 표시를 숨긴다 (Windows 콘솔에서 글자가 깨져 멈추는 일을 막는다)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self._ai_lock:
                    self.op.handle(trigger, t, **payload)
        except Exception as e:  # 흐름이 멈추지 않게 하고, 무슨 일이 있었는지 기록에 남긴다
            log_event("흐름 오류", "흐름", f"'{trigger}'을(를) 처리하다가 오류가 남", why=f"{type(e).__name__}: {str(e)[:200]}",
                      path=self.log_path, clock=self.clock(t), t=round(t, 1))
        finally:
            self.busy = None
