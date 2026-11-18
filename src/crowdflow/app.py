# 이 파일이 하는 일: 화면 세 개를 한 서버에서 띄운다 (Streamlit).
#   1) 관리자 화면(상황실): CCTV 영상과 사람 검출 상자, 약도(구역별 붐빔 색, 사람 점, 요원의 자리와 이동),
#      "지금 상황" 한 장, AI의 결정을 한 줄씩 보여 주는 목록(누르면 토론의 핵심과 근거가 펼쳐짐), 요원 현황, 글 지시 칸
#   2) 요원 화면(휴대폰): 자기에게 온 지시를 보고, 글로 답하고, "도착"을 누른다
#   3) 하루 정리 · 운영 지침 화면(주소 끝에 /review): 하루 정리 쓰기, 제안된 문장을 골라 지침에 더하기, 지침 읽기와
#      "어려운 날" 대본 켜고 끄기, 날짜별 숫자 견주기, 같은 상황을 지침 없이/읽고 물어보기 (D-64, 돌려 본 결과는 E-71)
#      **기본은 감춰져 있다 (D-68).** 설정 ui.review_screen을 true로 바꿔야 이 화면과 관리자 화면의 "하루 정리" 단추가 생긴다.
#   4) 결과 보고서 화면(주소 끝에 /report): 영상이 끝난 뒤 그 실행의 결과 보고서를 쓰고 보여 준다 (D-75). 3)의 정리와 같은 숫자와 글을
#      보고서 꼴(큰 숫자, 사건별 표, 현장에서 들어온 일, 확인 담당의 정리)로 보여 주고 파일로 남긴다. 운영 지침(RAG) 쪽은 여기에 없다.
#      설정 ui.result_report가 true이고 ui.review_screen이 false일 때 관리자 화면에 "결과 보고서" 단추가 생긴다 (기본).
#   화면들은 같은 시연 한 번(live.py)을 함께 본다. 관리자 화면과 요원 화면은 1초마다 스스로 새로 그린다.
# 해당 기능: 핵심 기능 1~4를 보여 주고 사람이 끼어드는 부분
# 해당 기준: C-16, C-21, C-31, C-43, C-44
# 근거: D-13·01의 11번 (관리자 화면에 넣을 것, "AI가 한 일 기록"을 가장 눈에 띄게), D-51 (대화방), D-52 (Streamlit),
#   D-55 (화면을 다시 만듦. 사용자 지시 2026-10-04: CCTV 영상과 검출 결과를 실시간으로, 결과는 간단히 보이고 누르면 토론의 핵심이 나오게,
#   상품처럼 보이게, 약도와 글꼴을 다듬고 요원 그림과 이동을 넣기), D-12·E-06 (녹화 영상임을 표시), E-07 (실행 기록을 보여 줄 것),
#   E-53 (Streamlit 1.65.0: st.fragment(run_every), st.cache_resource, st.navigation, st.html, st.expander의 key와 icon),
#   E-56 (화면을 그릴 때마다 하던 메모리 청소를 끔), D-58 (관리자의 "취소", 실시간 분석이 준비된 뒤에만 시작),
#   E-64 (CCTV 칸을 st.iframe에 넣은 까닭: 예전 영상 연결이 남아 새 영상이 뜨지 않던 것을 고침),
#   D-60 ("요원 대본" 스위치: 요원의 답과 도착을 대본이 대신 넣음. 켜 있는 동안 화면에 표시. 돌려 본 결과는 E-67),
#   D-75 (결과 보고서 화면. 사용자 지시 2026-10-05: 마지막에 결과 보고서 꼴로 보고서를 써 주는 기능을 보이게)
# 출처: AI 코딩 도구로 작성
#
# 실행: uv run streamlit run src/crowdflow/app.py   (관리자 화면은 주소 그대로, 요원 화면은 주소 끝에 /staff, 결과 보고서는 /report. 하루 정리 /review는 설정을 켰을 때만)
#   휴대폰에서 들어오려면: uv run streamlit run src/crowdflow/app.py --server.address 0.0.0.0
#   (Ollama가 켜져 있어야 AI 담당이 답한다. 실시간 분석 일꾼은 이 서버가 스스로 띄운다.)
from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src 폴더 (PYTHONPATH 없이도 돌게)

import streamlit as st  # noqa: E402

from crowdflow import chatroom as chat  # noqa: E402
from crowdflow.crowd_vision import load_yaml  # noqa: E402
from crowdflow.flow import APPROVE, ARRIVAL, CANCEL, FILL, MANAGER_TEXT, RELIEF, STAFF_TEXT, TICK  # noqa: E402
from crowdflow.live import ROOT, LiveOperation  # noqa: E402
from crowdflow.map_view import MAP_CSS, avatar_img, render_map, svg_img  # noqa: E402
from crowdflow.records import read_events  # noqa: E402
from crowdflow.review import REPORT_NOTICE, facts_text, field_notes, headline_numbers, incident_rows, result_report_markdown, writer_label  # noqa: E402

E = html.escape
DOING = {TICK: "AI 담당들이 상황을 판단하는 중", MANAGER_TEXT: "관리자의 글을 읽는 중", STAFF_TEXT: "요원의 답을 읽는 중",
         ARRIVAL: "도착을 처리하는 중", APPROVE: "승인을 처리하는 중", CANCEL: "지시를 취소하는 중"}
DISCLOSURE = "녹화 영상으로 재현 (Recorded Crowd Scenario) · 붐빔 기준과 요원은 시연용 설정 · 약도는 가상의 약도(축척 없음)"
STEPS = ["감지", "판단", "지시", "이동", "재확인", "종료"]
STEP_OF = {"대응 중": 1, "승인 대기": 1, "도착 대기": 3, "다시 확인 대기": 4, "관리자에게 넘김": 4, "관리자가 맡음": 4, "종료": 5}
VERDICT = {"close": "사건 종료", "send_more": "요원을 더 보냄", "keep_watching": "더 지켜봄", "notify_manager": "관리자에게 넘김"}
ROLES = {"효율 담당": ("eff", "효율"), "안전 담당": ("safe", "안전"), "지침 담당": ("guide", "지침"), "계획 담당": ("plan", "결정"), "규칙": ("tool", "규칙")}

CSS = """
:root{--bg:#0B1120;--panel:#111A2E;--panel2:#16213A;--line:#22304F;--text:#E6ECF7;--muted:#7C8BA8;--accent:#4F8CFF;
 --ok:#34D399;--warn:#FBBF24;--hot:#FB923C;--bad:#F87171;--num:Bahnschrift,'Pretendard Variable','Malgun Gothic',sans-serif}
.block-container{padding:1.0rem 1.5rem 1.2rem!important;max-width:100%!important}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stVerticalBlock"]{gap:.65rem}
.cf-top{display:flex;align-items:center;gap:18px;min-height:46px}
.cf-brand{display:flex;align-items:center;gap:10px;white-space:nowrap;flex:none;font-size:19px;font-weight:800;letter-spacing:-.01em;color:var(--text)}
.cf-brand small{font-size:12px;font-weight:600;color:var(--muted);letter-spacing:.02em;margin-left:2px}
.cf-brand b{color:var(--accent);font-weight:800}
.cf-clock{white-space:nowrap;flex:none;font:700 30px/1 var(--num);font-variant-numeric:tabular-nums;color:var(--text);margin-left:6px}
.cf-clock small{font-size:12px;color:var(--muted);font-weight:500;margin-left:8px;font-family:inherit}
.cf-chips{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto;justify-content:flex-end}
.cf-chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:5px 10px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:#B9C4DA;white-space:nowrap}
.cf-chip i{width:7px;height:7px;border-radius:50%;background:var(--muted);display:inline-block}
.cf-chip.ok i{background:var(--ok);box-shadow:0 0 0 3px rgba(52,211,153,.18)}.cf-chip.warn i{background:var(--warn)}.cf-chip.bad i{background:var(--bad)}
.cf-chip.busy{color:#C7D2FE;border-color:#3B4A7A;background:linear-gradient(90deg,#16213A,#1E2B52,#16213A);background-size:200% 100%;animation:cf-shine 1.4s linear infinite}
@keyframes cf-shine{to{background-position:-200% 0}}
.cf-h{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:700;letter-spacing:.06em;color:var(--muted);margin:2px 0 0}
.cf-h span{margin-left:auto;font-weight:500;letter-spacing:0;font-size:11.5px}
.cf-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.cf-camhead{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text);padding:0 2px}
.cf-camhead .tag{display:inline-grid;place-items:center;width:22px;height:20px;border-radius:6px;font-weight:800;font-size:12px;color:#0B1120}
.cf-camhead .st{color:var(--muted);font-size:11.5px}
.cf-camhead .n{margin-left:auto;font:700 19px/1 var(--num);font-variant-numeric:tabular-nums}
.cf-camhead .n small{font-size:10.5px;color:var(--muted);font-weight:500;margin-left:2px;font-family:inherit}
.cf-camhead.sm{gap:6px;font-size:12px;white-space:nowrap}.cf-camhead.sm b{overflow:hidden;text-overflow:ellipsis;min-width:0}
.cf-camhead.sm .n{font-size:16px}.cf-camhead.sm .cf-pill{font-size:10px;padding:1px 7px}
.cf-pill{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;border:1px solid currentColor;background:color-mix(in srgb,currentColor 14%,transparent)}
.cf-cam{position:relative;border-radius:12px;overflow:hidden;border:1px solid var(--line);background:#060A14;aspect-ratio:16/9}
@keyframes cf-blink{50%{opacity:.25}}
iframe[title*="CCTV 영상"]{aspect-ratio:16/9;height:auto!important;display:block}
.cf-cam.off{display:grid;place-items:center;text-align:center;color:var(--muted);font-size:12.5px;line-height:1.6;padding:10px}
.cf-legend{display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:var(--muted);padding:2px 4px}
.cf-legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.cf-hero{background:linear-gradient(135deg,#131D35,#101829);border:1px solid var(--line);border-radius:16px;padding:16px 18px;position:relative;overflow:hidden}
.cf-hero.hot{border-color:color-mix(in srgb,var(--c) 60%,var(--line));box-shadow:0 0 0 1px color-mix(in srgb,var(--c) 25%,transparent),0 10px 30px -18px var(--c)}
.cf-hero .k{font-size:11.5px;font-weight:700;letter-spacing:.06em;color:var(--muted)}
.cf-hero .t{display:flex;align-items:baseline;gap:10px;margin:6px 0 2px;font-size:24px;font-weight:800;letter-spacing:-.01em;color:var(--text)}
.cf-hero .t em{font-style:normal;color:var(--c)}
.cf-hero .t .n{margin-left:auto;font:700 30px/1 var(--num);color:var(--c);font-variant-numeric:tabular-nums}
.cf-hero .t .n small{font-size:12px;color:var(--muted);font-weight:500;margin-left:3px;font-family:inherit}
.cf-hero .d{font-size:13px;color:#B9C4DA;line-height:1.55}
.cf-steps{display:flex;gap:0;margin-top:12px}
.cf-steps div{flex:1;text-align:center;font-size:11px;color:var(--muted);position:relative;padding-top:14px}
.cf-steps div:before{content:"";position:absolute;top:3px;left:0;right:0;height:3px;background:var(--line)}
.cf-steps div:after{content:"";position:absolute;top:0;left:50%;width:9px;height:9px;margin-left:-4.5px;border-radius:50%;background:var(--line)}
.cf-steps div.done{color:#B9C4DA}.cf-steps div.done:before,.cf-steps div.done:after{background:var(--c)}
.cf-steps div.now{color:var(--text);font-weight:700}.cf-steps div.now:after{background:var(--c);box-shadow:0 0 0 4px color-mix(in srgb,var(--c) 30%,transparent);animation:cf-blink 1.2s steps(2) infinite}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line)!important;border-radius:12px!important}
[data-testid="stExpander"] summary{padding:.5rem .8rem!important;font-size:13.5px}
[data-testid="stExpander"] summary:hover{color:var(--text)}
[class*="st-key-tone-alert"] details{border-left:3px solid var(--bad)!important}
[class*="st-key-tone-warn"] details{border-left:3px solid var(--warn)!important}
[class*="st-key-tone-ok"] details{border-left:3px solid var(--ok)!important}
[class*="st-key-tone-ai"] details{border-left:3px solid var(--accent)!important}
[class*="st-key-tone-info"] details{border-left:3px solid #3B4A6B!important}
.cf-key{font-size:14.5px;font-weight:700;color:var(--text);margin:0 0 10px;line-height:1.45}
.cf-voice{display:grid;grid-template-columns:54px 1fr;gap:10px;padding:8px 0;border-top:1px solid var(--line)}
.cf-role{display:inline-block;font-size:11px;font-weight:800;text-align:center;border-radius:7px;padding:3px 0;height:22px}
.cf-role.eff{background:#0C3A5B;color:#7DD3FC}.cf-role.safe{background:#3B2A0A;color:#FCD34D}.cf-role.plan{background:#27215E;color:#C4B5FD}.cf-role.tool{background:#12324A;color:#93C5FD}.cf-role.chk{background:#123B2E;color:#6EE7B7}.cf-role.guide{background:#3A1E3F;color:#F0ABFC}
.cf-voice b{font-size:13.5px;color:var(--text)}.cf-voice b span{font-weight:600;color:var(--muted);font-size:11.5px;margin-left:6px}
.cf-voice p{margin:3px 0 0;font-size:12.5px;line-height:1.55;color:#9FB0CC}
.cf-opts{width:100%;border-collapse:collapse;margin:10px 0 4px;font-size:12.5px}
.cf-opts th{font-size:11px;font-weight:600;color:var(--muted);text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
.cf-opts td{padding:6px 8px;border-bottom:1px solid #1A2540;color:#C9D4E8}
.cf-opts tr.sel td{background:rgba(79,140,255,.10);color:var(--text);font-weight:600}
.cf-ex{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.cf-ex span{font-size:11.5px;padding:3px 9px;border-radius:8px;background:#141D33;border:1px solid var(--line);color:#A9B7D0}
.cf-ex b{color:#E6ECF7;font-weight:600}
.cf-foot{font-size:11.5px;color:var(--muted);margin-top:10px;line-height:1.5}
.cf-line{font-size:13px;color:#C9D4E8;line-height:1.6}.cf-line q{color:var(--text);font-weight:600;quotes:'“' '”'}
.cf-staff{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:6px}
.cf-s{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 4px 7px;text-align:center;position:relative;min-width:0}
.cf-s .av{border-radius:50%;box-shadow:0 0 0 2.5px var(--r),0 0 0 5px rgba(0,0,0,.25)}
.cf-s.rest .av{opacity:.45}
.cf-s .nm{font-size:12.5px;font-weight:700;color:var(--text);margin-top:6px}
.cf-s .zn{font-size:11px;color:var(--muted);margin-top:1px}
.cf-s .stt{display:inline-block;margin-top:5px;font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px;color:var(--r);border:1px solid var(--r);background:color-mix(in srgb,var(--r) 12%,transparent)}
.cf-s .br{font-size:10.5px;color:var(--muted);margin-top:4px}
.cf-raw{font:11.5px/1.6 Consolas,'Malgun Gothic',monospace;color:#9FB0CC;white-space:pre-wrap;word-break:break-all}
.cf-raw b{color:#E6ECF7;font-weight:600}
.cf-me{display:flex;align-items:center;gap:14px}.cf-me .av{border-radius:50%;box-shadow:0 0 0 3px var(--r)}
.cf-me h3{margin:0;font-size:20px;font-weight:800;color:var(--text)}.cf-me p{margin:2px 0 0;font-size:13px;color:var(--muted)}
.cf-msgs{display:flex;flex-direction:column;gap:8px}
.cf-msg{max-width:88%;padding:10px 13px;border-radius:14px;font-size:14.5px;line-height:1.5;background:var(--panel2);border:1px solid var(--line);color:var(--text)}
.cf-msg.me{align-self:flex-end;background:#1D3A7A;border-color:#2C4FA3}
.cf-msg small{display:block;font-size:11px;color:var(--muted);margin-bottom:3px}
.cf-msg.order{border-left:3px solid var(--warn)}
.cf-banner{padding:12px 14px;border-radius:12px;font-size:14px;font-weight:600;border:1px solid var(--line);background:var(--panel)}
.cf-ul{margin:4px 0 10px;padding-left:20px;font-size:13.5px;line-height:1.65;color:#C9D4E8}
.cf-opts td.k{color:var(--muted);font-size:12px}.cf-opts td.diff{color:var(--text);font-weight:700;background:rgba(79,140,255,.10)}
.cf-cl{display:grid;grid-template-columns:48px 1fr;gap:8px;padding:7px 0;border-top:1px solid var(--line);font-size:13px;line-height:1.55;color:#C9D4E8}
.cf-cl b{color:var(--accent);font:700 12px/1.9 var(--num)}.cf-cl.new b{color:#F0ABFC}.cf-cl small{display:block;color:var(--muted);font-size:11px}
.cf-rep{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 24px 16px}
.cf-rep .ttl{font-size:25px;font-weight:800;letter-spacing:-.01em;color:var(--text)}
.cf-rep .meta{font-size:12.5px;color:var(--muted);margin:5px 0 14px;line-height:1.5}
.cf-rep .sec{font-size:13px;font-weight:700;letter-spacing:.04em;color:var(--accent);margin:0 0 6px}
.cf-kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-bottom:16px}
.cf-kpi{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px 12px 9px}
.cf-kpi .l{font-size:12px;font-weight:600;color:var(--muted)}
.cf-kpi .v{font:700 27px/1.25 var(--num);font-variant-numeric:tabular-nums;color:var(--text);margin-top:2px}
.cf-kpi .s{font-size:11.5px;color:#9FB0CC;line-height:1.4;margin-top:2px}
.cf-rep .cols{display:grid;grid-template-columns:1.5fr 1fr;gap:26px}
.cf-rep .cf-opts{margin:2px 0 14px;font-size:13px}.cf-rep .cf-opts td b{color:var(--text)}
.cf-rep .cf-ul{margin:2px 0 12px}
.cf-rep .verdict{display:inline-block;font-size:13px;font-weight:800;padding:2px 12px;border-radius:999px;border:1px solid currentColor;margin-left:6px}
"""


@st.cache_resource(show_spinner="행사장 자료를 읽는 중…")
def engine() -> LiveOperation:
    """서버에 하나만 있는 시연. 모든 화면(관리자, 요원들의 휴대폰)이 이것을 함께 쓴다."""
    # 시험할 때만 쓰는 값: 기록을 다른 폴더에 두고(켜져 있는 서버의 기록을 지우지 않게), AI 대신 규칙으로 돌리고, 실시간 분석을 끈다
    out_dir = os.environ.get("CROWDFLOW_LIVE_DIR")
    vision = load_yaml(ROOT / "config" / "settings.yaml")["vision_live"]["enabled"] and os.environ.get("CROWDFLOW_VISION") != "off"
    host = "127.0.0.1" if st.get_option("server.address") in ("localhost", "127.0.0.1") else "0.0.0.0"
    return LiveOperation(use_rule=os.environ.get("CROWDFLOW_BRAIN") == "rule", out_dir=Path(out_dir) if out_dir else None,
                         vision=vision, vision_host=host)


def colors(eng: LiveOperation) -> list[str]:
    return eng.settings["ui"]["level_colors"]


def place(eng: LiveOperation, zone: str) -> str:
    return f"{zone}구역" if zone in eng.venue.zones else zone


def main_incident(eng: LiveOperation) -> dict | None:
    """열려 있는 사건 가운데 화면 맨 앞에 보여 줄 것: 단계가 가장 높은 것, 같으면 먼저 열린 것."""
    names = [lv["name"] for lv in eng.settings["crowd_levels"]["levels"]]
    return max(eng.op.state.incidents.values(), key=lambda i: (names.index(i["level"]) if i["level"] in names else -1, -i["id"]), default=None)


def static_sig(eng: LiveOperation) -> tuple:
    """화면에서 한 번만 그리는 부분(영상, 버튼)이 기대는 값. 이 값이 바뀌면 화면 전체를 다시 그린다."""
    moving = tuple(sorted(s["id"] for s in eng.op.state.staff if s["status"] == "이동 중"))  # '취소' 단추가 달라져야 하는지
    return (eng.vision_info()["live"], (main_incident(eng) or {}).get("zone") or eng.settings["ui"]["focus_zone"], eng.started, eng._gen, moving,
            eng.staff_script)  # '요원 대본' 단추의 글(켜기/끄기)이 달라져야 하는지


# ---- 결정 목록: 기록(events)을 "한 줄 결과 + 눌렀을 때의 근거"로 묶는다 ---------------------------------------
def build_cards(events: list[dict]) -> tuple[list[dict], dict]:
    cards: list[dict] = []
    plans: dict[str, dict] = {}
    detect = manager_text = recheck = None
    replies: dict[str, dict] = {}
    for n, e in enumerate(events):
        k, pid = e["kind"], e.get("plan_id")
        plan = plans.setdefault(pid, {"options": [], "debate": [], "orders": [], "detect": None, "guide": None}) if pid else None
        if k == "붐빔 감지":
            detect = e
        elif k == "지침 찾기":  # 토론 전에 운영 지침에서 읽어 온 조항 (D-64)
            plan["guide"] = e
        elif k == "배치 계산":
            plan["options"].append(e)
        elif k == "토론":
            plan["debate"].append(e)
        elif k == "계획":
            plan["plan"], plan["detect"], detect = e, detect, None
            cards.append({"type": "plan", "n": n, "e": e, "pid": pid})
        elif k == "관리자에게 넘김":
            cards.append({"type": "handover", "n": n, "e": e, "detect": detect, "pid": pid})
            detect = None
        elif k == "지시 보냄":
            if e.get("purpose") == RELIEF:
                for c in reversed(cards):
                    if c["type"] == "relief" and c["e"].get("staff") == e.get("staff"):
                        c["order"] = e
                        break
            elif plan is not None:
                plan["orders"].append(e)
        elif k in (RELIEF, "교대 요원 없음"):
            cards.append({"type": "relief", "n": n, "e": e})
        elif k in (FILL, "최소 인원 모자람"):  # 관리자가 정한 최소 인원을 채우러 보냄 / 보낼 요원이 모자람 (D-62)
            cards.append({"type": "fill", "n": n, "e": e})
        elif k == "관리자 글":
            manager_text = e
        elif k == "글 읽기" and "applied" in e:
            if not e["applied"]:
                cards.append({"type": "unread", "n": n, "e": e, "text": manager_text})
        elif k == "조건 바꿈":
            cards.append({"type": "rule", "n": n, "e": e, "text": manager_text})
        elif k == "요원 글":
            replies[e.get("staff", "")] = e
        elif k == "글 읽기" and "can_go" in e:
            cards.append({"type": "reply", "n": n, "e": e, "text": replies.get(e.get("staff", "")), "excluded": False})
        elif k == "요원 제외":
            for c in reversed(cards):
                if c["type"] == "reply" and c["e"].get("staff") == e.get("staff"):
                    c["excluded"] = True
                    break
        elif k == "도착":
            cards.append({"type": "arrival", "n": n, "e": e})
        elif k == "다시 세기":
            recheck = e
        elif k == "결과 확인":
            cards.append({"type": "verdict", "n": n, "e": e, "recheck": recheck})
        elif k in ("휴게 시작", "휴게 중", "휴게 끝"):
            cards.append({"type": "break", "n": n, "e": e})
        elif k in ("AI 오류", "흐름 오류", "AI 답 바로잡음", "관리자 승인", "지시 취소", "복귀 지시", "관리자에게 알림", "지침 찾기 오류", "하루 정리", "지침 더함"):
            cards.append({"type": "system", "n": n, "e": e})
    return cards, plans


def voice(role_cls: str, role: str, head: str, tail: str = "", body: str = "") -> str:
    return (f'<div class="cf-voice"><span class="cf-role {role_cls}">{E(role)}</span><div><b>{E(head)}<span>{E(tail)}</span></b>'
            + (f"<p>{E(body)}</p>" if body else "") + "</div></div>")


def excluded_chips(items: list[dict]) -> str:
    rows = [f"<span><b>{E(x['name'])}</b> {E(x['reason'])}</span>" for x in items if "채워짐" not in x["reason"]]
    return ('<div class="cf-foot" style="margin-top:12px">보내지 않은 요원과 까닭</div><div class="cf-ex">' + "".join(rows) + "</div>") if rows else ""


def card_view(card: dict, plans: dict, eng: LiveOperation) -> tuple[str, str, str, str]:
    """카드 하나를 (색 종류, 아이콘, 한 줄 제목, 펼쳤을 때의 내용)으로 바꾼다."""
    e, t = card["e"], card["type"]
    if t == "plan":
        p = plans[card["pid"]]
        who = ", ".join(o["what"].split("에게")[0] for o in p["orders"])
        act = f"{who} 출동" if who else "더 보내지 않고 지켜봄" if e.get("chosen") == 0 else "배치안 승인 대기"
        d = p["detect"]
        title = (f"{d['zone']}구역 {d['level']} 감지 · 평균 {d['count_avg']:.0f}명 → {act}" if d else f"다시 계산 → {act}")
        chosen = e.get("chosen", 1)
        body = f'<div class="cf-key">{E(e.get("headline") or e["what"])}</div>'
        for v in p["debate"]:
            cls, short = ROLES.get(v["who"], ("tool", v["who"][:2]))
            tail = "해당 조항 없음" if v["who"] == "지침 담당" and not v.get("option") else f"선택지 {v.get('option', '')}"
            body += voice(cls, short, v.get("point") or v["what"], tail, v["why"])
        if p.get("guide"):  # 지침 찾기 도구가 읽어 온 조항. 지침 담당이 해당한다고 본 조항에 ✓
            g = p["guide"]
            used = {i for v in p["debate"] if v["who"] == "지침 담당" for i in (v.get("clauses") or [])}
            rows = "".join(f"<span><b>{'✓ ' if c['id'] in used else ''}{E(c['id'])}</b> {E(c['text'])}"
                           + (f" · 가까운 정도 {c['score']:.2f}" if c.get("score") is not None else "") + f" · {E(c['source'])}</span>" for c in g.get("clauses", []))
            body += (f'<div class="cf-foot" style="margin-top:12px">운영 지침에서 읽어 온 조항 · {E(g.get("mode", ""))} (조항 {g.get("total", "")}개 가운데 {len(g.get("clauses", []))}개)</div>'
                     f'<div class="cf-ex">{rows}</div>')
        if not p["debate"]:
            body += voice("plan", "결정", e["what"], "", "" if e["who"] == "흐름" else "토론할 선택지가 하나뿐이었습니다.")
        if p["options"]:
            body += '<table class="cf-opts"><tr><th></th><th>선택지</th><th>보내는 요원</th><th>이동</th><th>예비 요원</th></tr>'
            for o in p["options"]:
                f = o.get("facts", {})
                trav = "정해지지 않음" if f.get("longest_travel_seconds") is None else f"{f['longest_travel_seconds']:g}초"
                body += (f'<tr class="{"sel" if o.get("option") == chosen else ""}"><td>{"✓" if o.get("option") == chosen else ""}</td><td>{o.get("option")}. {E(o.get("title", ""))}</td>'
                         f'<td>{E(", ".join(m["name"] for m in o.get("moves", [])) or "없음")}</td><td>{trav}</td><td>{"씀" if f.get("uses_reserve") else "남김"}</td></tr>')
            body += "</table>" + excluded_chips(p["options"][0].get("excluded", []))
        body += '<div class="cf-foot">선택지는 배치 계산 도구가 만듭니다(휴게와 최소 인원을 지키는 안만). AI 담당들은 그 안에서만 고릅니다.'
        models = {v["who"]: v["model"] for v in p["debate"] if v.get("model")}  # 어느 모델이 낸 말인지 (D-65)
        if len(set(models.values())) > 1:
            body += (f' 효율·안전 의견은 가벼운 모델({E(models.get("효율 담당", ""))})이 내고, 결정은 큰 모델({E(models.get("계획 담당", ""))})이 합니다.')
        body += "</div>"
        return "ai", ":material/campaign:", title, body
    if t == "handover":
        d = card.get("detect")
        title = (f"{d['zone']}구역 {d['level']} 감지 → " if d else "") + "보낼 수 있는 요원 없음 · 관리자 확인 필요"
        opts = plans.get(card.get("pid"), {}).get("options") or [{}]
        return "alert", ":material/warning:", title, ('<div class="cf-key">조건(휴게, 최소 인원)을 지키면서 보낼 수 있는 요원이 없어 관리자에게 넘겼습니다.</div>'
                                                     + excluded_chips(opts[0].get("excluded", [])))
    if t == "relief":
        if e["kind"] == RELIEF:
            body = f'<div class="cf-key">{E(e["what"])}</div>' + voice("tool", "계산", e["why"].split(";")[0], "", "휴게는 붐벼도 미루지 않습니다. 쉬는 요원의 자리를 바꿔 줄 요원이 휴게 전에 도착하도록 보냅니다.") + excluded_chips(e.get("excluded", []))
            return "warn", ":material/swap_horiz:", "휴게 교대 · " + e["what"].replace("이(가)", "").replace("을(를)", "").replace("교대 요원으로 보냄", "교대"), body
        return "alert", ":material/warning:", e["what"] + " · 관리자 확인 필요", f'<div class="cf-key">교대할 요원을 구하지 못했습니다. 휴게는 미루지 않습니다.</div>' + excluded_chips(e.get("excluded", []))
    if t == "fill":
        names = ", ".join(m["name"] for m in e.get("moves", []))
        if e["kind"] == FILL:
            body = f'<div class="cf-key">{E(e["what"])}</div>' + \
                voice("tool", "계산", e["why"].split(";")[0], "", "누구를 보낼지는 배치 계산 도구가 골랐습니다. 휴게와 다른 구역의 최소 인원은 그대로 지킵니다.") + \
                excluded_chips(e.get("excluded", []))
            return "ok", ":material/group_add:", f"최소 인원 채우기 · {e['zone']}구역에 {names} 출동", body
        return "alert", ":material/warning:", e["what"] + " · 관리자 확인 필요", \
            '<div class="cf-key">조건을 지키면서 보낼 수 있는 요원이 없습니다. 휴게와 다른 구역의 최소 인원은 깨지 않습니다.</div>' + excluded_chips(e.get("excluded", []))
    if t == "rule":
        said = (card.get("text") or {}).get("what", "")
        have, need = e.get("on_duty"), e["min_staff"]
        note = "이 구역의 요원은 이 인원 밑으로 빼 가지 않습니다."
        if have is not None:  # 지금 인원이 모자라면 채울 요원을 찾는다 (D-62). 결과는 바로 아래 줄에 나온다
            note = (f"지금 근무 인원은 {have}명이라 {need - have}명이 모자랍니다. 채울 요원을 찾습니다. " if have < need else f"지금 근무 인원은 {have}명입니다. ") + note
        body = f'<div class="cf-line">관리자: <q>{E(said.strip(chr(34)))}</q></div>' + voice("plan", "소통", e["what"], "", note)
        return "ok", ":material/edit_note:", f"지시 반영 · {e['zone']}구역 최소 인원 {e.get('before', '')}→{e['min_staff']}명", body
    if t == "unread":
        said = (card.get("text") or {}).get("what", "")
        return "warn", ":material/help:", "지시를 조건으로 바꾸지 않음 · 되물음", f'<div class="cf-line">관리자: <q>{E(said.strip(chr(34)))}</q></div>' + voice("plan", "소통", e["why"])
    if t == "reply":
        name = e["what"].split("의 답")[0]
        said = (card.get("text") or {}).get("what", "").strip('"')
        head = {"yes": f"{name} 출동 확인", "no": f"{name} 출동 불가", "unclear": f"{name}의 답을 이해하지 못함 · 되물음"}[e["can_go"]]
        if e.get("injury"):
            head += " · 다친 사람 보고"
        by_script = (card.get("text") or {}).get("scripted")  # 사람이 아니라 대본이 넣은 답 (D-60)
        if by_script:
            head += " (대본)"
        body = f'<div class="cf-line">{E(name)}{" (대본이 대신 넣음)" if by_script else ""}: <q>{E(said)}</q></div>' + \
            voice("plan", "소통", e["what"], "", e["why"] if e["why"] != said else "")
        if e.get("injury"):
            body += voice("chk", "알림", "다친 사람이 있다는 보고를 관리자에게 따로 알렸습니다.")
        if card["excluded"]:
            body += voice("tool", "흐름", f"{name}을(를) 빼고 다시 계산했습니다.")
        return ("alert" if e.get("injury") else "warn" if e["can_go"] != "yes" else "ok"), ":material/chat:", head, body
    if t == "arrival":
        pressed = "대본이 요원 대신 ‘도착’을 눌렀습니다 (요원 역할: 대본)." if e.get("scripted") else "요원이 화면에서 ‘도착’을 눌렀습니다."
        return "ok", ":material/where_to_vote:", f"{e['who']} {e.get('zone', '')}구역 도착" + (f" · {e['purpose']}" if e.get("purpose") else "") + \
            (" (대본)" if e.get("scripted") else ""), f'<div class="cf-line">{pressed} 붐빔 대응으로 보낸 요원이 모두 도착하면 잠시 뒤 다시 셉니다.</div>'
    if t == "verdict":
        r = card.get("recheck") or {}
        closed = e.get("decision") == "close"
        title = f"다시 세기 · {e.get('zone', '')}구역 {e.get('count_avg', 0):.0f}명 {e.get('level', '')} → {VERDICT.get(e.get('decision'), e['what'])}"
        body = voice("tool", "세기", f"보내기 전 평균 {r.get('before', '?')}명 → 지금 평균 {r.get('count_avg', '?')}명 ({r.get('level', '')})") + voice("chk", "확인", VERDICT.get(e.get("decision"), e["what"]), "", e["why"])
        if closed:
            body += '<div class="cf-foot">녹화 영상이라, 요원을 보내서 사람이 줄었다고 볼 수는 없습니다.</div>'
        return ("ok" if closed else "warn"), (":material/check_circle:" if closed else ":material/query_stats:"), title, body
    if t == "break":
        return "info", ":material/coffee:", e["what"], f'<div class="cf-line">{E(e["why"])}</div>'
    tone, icon = {"복귀 지시": ("ok", ":material/undo:"), "지시 취소": ("warn", ":material/cancel:"), "관리자에게 알림": ("alert", ":material/notifications_active:")}.get(
        e["kind"], ("alert" if "오류" in e["kind"] else "info", ":material/info:"))
    return tone, icon, (e["what"] if e["kind"] in ("복귀 지시", "지시 취소") else f"{e['kind']} · {e['what']}"), f'<div class="cf-line">{E(e["why"])}</div>'


# ---- 관리자 화면: 1초마다 새로 그리는 부분 ---------------------------------------------------------------
def top_html(eng: LiveOperation) -> str:
    info, manual = eng.vision_info(), eng.settings["operation"]["approval_mode"] == "manual"
    chips = ['<span class="cf-chip"><i></i>녹화 영상 재현</span>']
    if info["live"]:
        chips.append(f'<span class="cf-chip ok"><i></i>GPU 실시간 분석 · 초당 {info["rate"] or 0:.0f}회</span>')
    else:
        chips.append('<span class="cf-chip warn"><i></i>분석 일꾼 켜는 중</span>' if info["on"] else '<span class="cf-chip"><i></i>미리 분석한 결과 사용</span>')
    chips.append({None: '<span class="cf-chip warn"><i></i>AI 모델 올리는 중</span>', True: f'<span class="cf-chip ok"><i></i>{E(eng.brain.name)} 준비됨</span>',
                  False: '<span class="cf-chip bad"><i></i>AI 서버에 닿지 않음 · 규칙으로 대신</span>'}[eng.model_ready])
    chips.append(f'<span class="cf-chip"><i></i>{"관리자 승인 뒤 실행" if manual else "자동 실행 · 관리자에게 알림"}</span>')
    if eng.staff_script:  # 요원의 답과 도착을 사람이 아니라 대본이 넣는 중임을 늘 보여 준다 (D-60, E-06)
        chips.append(f'<span class="cf-chip warn"><i></i>요원 역할: 대본{" · 어려운 날" if eng.hard_day else ""}</span>')
    if eng.guideline_on:  # 토론에서 운영 지침을 읽히는 중 (D-64)
        chips.append(f'<span class="cf-chip ok"><i></i>지침 읽기 · 조항 {len(eng.clauses())}개</span>')
    busy = eng.busy
    if busy and (busy[0] != TICK or time.monotonic() - busy[1] > 1.0):
        chips.append(f'<span class="cf-chip busy">{DOING.get(busy[0], "처리하는 중")}…</span>')
    sub = "‘시작’을 누르면 영상 시계가 흐릅니다" if not eng.started else "영상이 끝났습니다" if eng.finished else f"영상 {eng.t:.0f} / {eng.total:.0f}초"
    logo = svg_img('<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 26 26"><circle cx="13" cy="13" r="12" fill="none" stroke="#4F8CFF" stroke-width="2"/>'
                   '<circle cx="13" cy="13" r="6.5" fill="none" stroke="#4F8CFF" stroke-width="2" opacity=".55"/><circle cx="13" cy="13" r="2.4" fill="#4F8CFF"/></svg>', "cf-logo")
    return (f'<div class="cf-top"><div class="cf-brand">{logo}CrowdFlow <b>Agent</b><small>행사 안전 상황실</small></div>'
            f'<div class="cf-clock">{eng.clock()}<small>{sub}</small></div><div class="cf-chips">{"".join(chips)}</div></div>')


def cam_head(eng: LiveOperation, k: str, s: dict, small: bool = False) -> str:
    """CCTV 칸 위의 한 줄. 작은 칸에서는 자리가 좁아 거리 이름을 뺀다."""
    c, z = colors(eng)[s["level_index"]], eng.venue.zones[k]
    street = "" if small else f'<span class="st">{E(z.street)}</span>'
    return (f'<div class="cf-camhead{" sm" if small else ""}"><span class="tag" style="background:{c}">{E(k)}</span><b>{E(z.name)}</b>{street}'
            f'<span class="n" style="color:{c}">{s["count_avg"]:.0f}<small>명</small></span><span class="cf-pill" style="color:{c}">{E(s["level_name"])}</span></div>')


# CCTV 한 칸을 그리는 작은 문서. st.iframe 안에 넣는다.
# 까닭: 영상(MJPEG)은 끝나지 않는 연결이다. 화면이 칸을 새로 그릴 때 예전 <img>의 연결이 바로 끊기지 않고 남아서,
#       브라우저가 한 주소에 열어 주는 연결(6개)을 다 채우면 새 영상이 뜨지 않는다 (E-63에서 직접 확인).
#       따로 떨어진 문서는 바뀌거나 사라질 때 브라우저가 그 안의 연결을 바로 끊는다.
CAM_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">__FONT__<style>
html,body{margin:0;padding:0;background:__BG__;overflow:hidden}
.cam{position:relative;box-sizing:border-box;aspect-ratio:16/9;border-radius:12px;overflow:hidden;border:1px solid #22304F;background:#060A14}
.cam img{width:100%;height:100%;object-fit:cover;display:block}
.live{position:absolute;left:8px;top:8px;font:700 10.5px/1.5 'Pretendard Variable','Malgun Gothic',sans-serif;letter-spacing:.06em;padding:3px 8px;border-radius:6px;
 background:rgba(6,10,20,.72);color:#E6ECF7;display:flex;align-items:center;gap:6px;white-space:nowrap}
.live i{width:7px;height:7px;border-radius:50%;background:#EF4444;animation:b 1.2s steps(2) infinite}
@keyframes b{50%{opacity:.25}}
</style></head><body><div class="cam"><img src="__SRC__" alt="__ALT__"><div class="live"><i></i>__LABEL__</div></div></body></html>"""


def cam_page(src: str, label: str, alt: str) -> str:
    font = st.get_option("theme.font") or ""  # 화면과 같은 글꼴을 쓴다 (.streamlit/config.toml 한 곳에서 정함)
    url = font.split(":", 1)[1].split(",")[0].strip() if ":" in font.split(",")[0] else ""
    link = f'<link rel="stylesheet" href="{E(url)}">' if url.startswith("http") else ""
    return (CAM_PAGE.replace("__FONT__", link).replace("__BG__", st.get_option("theme.backgroundColor") or "#0B1120")
            .replace("__SRC__", E(src)).replace("__ALT__", E(alt)).replace("__LABEL__", E(label)))


def hero_html(eng: LiveOperation, status: dict, staff: list[dict]) -> str:
    state, cs = eng.op.state, colors(eng)
    inc = main_incident(eng)
    names = {s["id"]: s["name"] for s in staff}
    if not eng.started:
        return ('<div class="cf-hero" style="--c:#4F8CFF"><div class="k">지금 상황</div><div class="t">시연 대기 중</div>'
                '<div class="d">‘시작’을 누르면 CCTV 영상이 흐르고, AI가 구역별 사람 수를 보며 스스로 대응합니다.</div></div>')
    if not inc:
        last = state.closed[-1] if state.closed else None
        d = (f"최근 사건: {last['zone']}구역 평균 {last['before_avg']}명 → {last['after_avg']}명으로 내려가 종료했습니다." if last
             else "요원을 보내는 기준(" + eng.settings["crowd_levels"]["dispatch_from"] + ")에 닿은 구역이 없습니다.")
        return (f'<div class="cf-hero" style="--c:{cs[0]}"><div class="k">지금 상황</div><div class="t"><em>●</em> 모든 구역 평상</div><div class="d">{E(d)}</div></div>')
    s = status[inc["zone"]]
    c = cs[max(s["level_index"], 1)]
    step = STEP_OF.get(inc["status"], 1)
    lines = [f"{x['name']} 이동 중 ({x['purpose'] or '이동'})" for x in staff if x["moving"] and x["target"] == inc["zone"]]
    lines += [f"{names[i]} 도착" for i in inc["arrived"]]
    others = [f"{i['zone']}구역 {i['status']}" for i in state.incidents.values() if i is not inc]
    if inc["status"] == "관리자에게 넘김":
        lines.insert(0, "AI가 더 할 수 있는 일이 없어 관리자에게 넘겼습니다")
    elif inc["status"] == "승인 대기":
        lines.insert(0, "배치안이 관리자의 승인을 기다립니다")
    elif inc["status"] == "다시 확인 대기":
        lines.append(f"곧 다시 셉니다 (다시 확인 {inc['rechecks']}회)")
    elif inc["status"] == "대응 중":
        lines.insert(0, "AI 담당들이 대응을 정하는 중")
    if others:
        lines.append("다른 사건: " + ", ".join(others))
    steps = "".join(f'<div class="{"done" if i < step else "now" if i == step else ""}">{n}</div>' for i, n in enumerate(STEPS))
    return (f'<div class="cf-hero hot" style="--c:{c}"><div class="k">지금 상황 · 사건 {inc["id"]}</div>'
            f'<div class="t"><em>{E(inc["zone"])}구역 {E(s["level_name"])}</em><span class="n">{s["count_avg"]:.0f}<small>명 (5초 평균)</small></span></div>'
            f'<div class="d">{E(" · ".join(lines) or inc["status"])}</div><div class="cf-steps">{steps}</div></div>')


def staff_html(eng: LiveOperation, staff: list[dict]) -> str:
    ring = {"근무": "#38BDF8", "이동": "#FBBF24", "휴게": "#64748B"}
    out = []
    for s in staff:
        r = ring[s["state"]]
        where = f"{place(eng, s['zone'])} → {place(eng, s['target'])}" if s["moving"] else place(eng, s["zone"])
        label = "이동 중" if s["moving"] else "휴게 중" if s["state"] == "휴게" else "지금 못 감" if s["blocked"] else "곧 휴게" if s["soon"] else "근무"
        if s["blocked"] and not s["moving"]:
            r = "#F87171"
        elif s["soon"] and s["state"] == "근무":
            r = "#FBBF24"
        br = f"휴게 {s['rest_left']:.0f}분 남음" if s["state"] == "휴게" else f"휴게 {s['next_break']}"
        out.append(f'<div class="cf-s {"rest" if s["state"] == "휴게" else ""}" style="--r:{r}" title="{E(s["reason"])}">{avatar_img(s["id"], 42)}'
                   f'<div class="nm">{E(s["name"])}</div><div class="zn">{E(where)}</div><div class="stt">{label}</div><div class="br">{br}</div></div>')
    return '<div class="cf-staff">' + "".join(out) + "</div>"


UI = load_yaml(ROOT / "config" / "settings.yaml")["ui"]


@st.fragment(run_every=UI["map_refresh_seconds"])
def manager_fast(slots: dict) -> None:
    """자주 새로 그리는 부분: 시계, CCTV 위의 사람 수, 약도의 사람 점."""
    eng = engine()
    points, status = eng.view()
    slots["top"].html(top_html(eng))
    for k, (slot, small) in slots["cams"].items():
        slot.html(cam_head(eng, k, status[k], small))
    slots["map"].html(svg_img(render_map(eng.venue, status, points, eng.staff_view(), colors(eng), eng.settings["staffing"]["standby_zone"], t=time.monotonic()), "cf-map"))


@st.fragment(run_every=UI["refresh_seconds"])
def manager_live(slots: dict) -> None:
    eng = engine()
    if st.session_state.get("static_sig") != static_sig(eng):  # 영상이나 버튼이 달라져야 하면 화면 전체를 다시 그린다
        st.rerun()
    state = eng.op.state
    status = eng.status_at(eng.t)
    staff = eng.staff_view()
    slots["hero"].html(hero_html(eng, status, staff))
    slots["staff"].html(staff_html(eng, staff))

    events = read_events(eng.log_path)
    cards, plans = build_cards(events)
    manual = eng.settings["operation"]["approval_mode"] == "manual"
    with slots["feed"].container():
        if manual and (state.pending_instruction or any(i.get("proposal") for i in state.incidents.values())):
            if st.button("배치안 승인", type="primary", key="approve", width="stretch"):
                eng.submit(APPROVE)
        if not cards:
            st.html('<div class="cf-line" style="color:var(--muted);padding:6px 4px">아직 결정이 없습니다. AI가 무언가를 정하면 여기에 한 줄씩 쌓입니다. 줄을 누르면 근거가 펼쳐집니다.</div>')
        for card in reversed(cards[-40:]):
            tone, icon, title, body = card_view(card, plans, eng)
            with st.container(key=f"tone-{tone}-{card['n']}"):
                with st.expander(f":gray[{events[card['n']].get('clock', '')}]  **{title}**", icon=icon, key=f"card-{card['n']}"):
                    st.html(body)
    with slots["raw"].container():
        with st.expander("전체 실행 기록 (원본 그대로) · 대화방", icon=":material/receipt_long:", key="raw"):
            lines = [f"<b>{E(e.get('clock', ''))} [{E(e['who'])}] {E(e['kind'])}</b> {E(e['what'])}" + (f"  ← {E(e['why'])}" if e.get("why") else "") for e in events[-80:]]
            talk = [f"<b>{E(m.clock)} {E(m.sender)} → {E(m.to)} [{E(m.kind)}]</b> {E(m.text)}" for m in eng.room.all()]
            st.html('<div class="cf-raw">' + "\n".join(lines or ["(기록 없음)"]) + "\n\n— 대화방 —\n" + "\n".join(talk or ["(글 없음)"]) + "</div>")


@st.fragment
def manager_form() -> None:
    """관리자의 글 지시 칸. 따로 떨어진 부분(fragment)으로 두어, ‘보내기’를 눌러도 이 칸만 다시 그리게 한다.

    전에는 ‘보내기’가 화면 전체를 다시 그리게 했다. 시연 영상을 찍다가 ‘보내기’를 누른 순간 화면이 통째로 빈 일이 한 번 있었고(E-78),
    전에도 단추를 누를 때 0.3초마다 도는 부분과 겹쳐 화면이 빈 일이 있었다(E-64). 다시 일으키지는 못해 원인을 확정하지 못했지만,
    화면 전체를 다시 그릴 일을 하나 줄여 둔다. 글을 받아 처리하는 일(eng.submit)은 그대로다.
    """
    eng = engine()
    with st.form("manager_text", clear_on_submit=True, border=False):
        f1, f2 = st.columns([5, 1], vertical_alignment="bottom")
        text = f1.text_input("관리자 지시", placeholder="글로 지시하기 · 예: A구역은 2명 유지해", label_visibility="collapsed")
        if f2.form_submit_button("보내기", width="stretch") and text.strip():
            eng.submit(MANAGER_TEXT, text=text.strip())
            st.toast("지시를 전달했습니다. 소통 담당이 읽습니다.")


def manager_page() -> None:
    eng = engine()
    st.session_state["static_sig"] = sig = static_sig(eng)
    live, focus = sig[0], sig[1]
    if "review" in PAGES:  # 결과 보고서(ui.result_report, D-75)나 하루 정리 화면(ui.review_screen, D-68)을 보여 주는 설정일 때만 단추가 있다
        top, b_rev, b0, b1, b2 = st.columns([10.0, 1.5, 1.9, 1.1, 1.6], vertical_alignment="center")
        # 영상이 끝난 뒤 "처음부터 다시"를 누르기 전에 보고서(정리)를 쓴다 (다시 하면 이번 실행의 기록이 지워진다)
        b_rev.page_link(PAGES["review"], label="하루 정리" if UI.get("review_screen") else "결과 보고서", icon=":material/summarize:", width="stretch")
    else:
        top, b0, b1, b2 = st.columns([11.5, 1.9, 1.1, 1.6], vertical_alignment="center")
    slots: dict = {"top": top.empty(), "cams": {}}
    # 요원 역할을 대본이 대신하기 (D-60): 혼자 시연하거나 연습할 때 켠다. 켜 있는 동안 위쪽에 "요원 역할: 대본"이 보인다.
    # 단추의 글은 "누르면 할 일"이다. 처음에는 스위치(toggle)로 만들었는데, 관리자 화면을 두 군데서 열어 두면
    # 다른 화면에 남아 있던 예전 값이 방금 켠 것을 도로 꺼 버렸다 (E-69). 단추는 값을 들고 있지 않아 그런 일이 없다.
    script_on = sig[5]
    if b0.button("요원 대본 끄기" if script_on else "요원 대본 켜기", width="stretch",
                 help="켜면 요원의 답과 ‘도착’을 사람 대신 대본이 넣습니다. 처음 지시받은 요원은 못 간다고 답하고, 그 밖의 요원은 이동 시간이 지나면 도착을 누릅니다."):
        eng.staff_script = not script_on
        st.rerun()
    if b1.button("시작", type="primary", disabled=eng.started, width="stretch"):
        if eng.model_ready is None:  # 모델을 올리는 중에 시작하면 AI의 첫 답이 1분쯤 늦어진다 (E-54)
            st.toast("AI 모델을 올리는 중입니다. ‘준비됨’이 보이면 시작해 주세요.")
        elif eng.vision_info()["on"] and not live:  # 실시간 분석이 준비되기 전에 시작하면 미리 분석해 둔 결과로 돌게 된다
            st.toast("실시간 분석 일꾼을 켜는 중입니다. ‘GPU 실시간 분석’이 보이면 시작해 주세요.")
        else:
            eng.start()
            st.rerun()
    if b2.button("처음부터 다시", width="stretch"):
        with st.spinner("처음 상태로 돌리는 중…"):
            eng.reset()
        st.rerun()

    col_cam, col_map, col_side = st.columns([4.7, 4.1, 5.2], gap="medium")
    with col_cam:
        head, tog = st.columns([3, 2], vertical_alignment="center")
        head.html('<div class="cf-h">CCTV · 사람 검출</div>')
        boxes = tog.toggle("검출 상자 표시", value=True, key="boxes")
        host = (st.context.headers.get("host") or "localhost").split(":")[0]
        info = eng.vision_info()

        def cam(k: str, small: bool = False) -> None:
            slots["cams"][k] = (st.empty(), small)
            if live:
                cctv = eng.venue.zones[k].cctv.id
                st.iframe(cam_page(f'http://{host}:{info["port"]}/stream/{k}{"" if boxes else "?boxes=0"}', cctv if small else f"{cctv} · 녹화 영상 재현", f"CCTV {k}"),
                          height="content", alt=f"{k}구역 CCTV 영상 (녹화 영상 재현)")
            else:
                msg = "실시간 분석 일꾼을 켜는 중입니다…" if info["on"] else "실시간 분석이 꺼져 있습니다."
                st.html(f'<div class="cf-cam off"><div>{msg}<br>사람 수는 미리 분석해 둔 결과로 표시합니다.</div></div>')

        keys = list(eng.venue.zones)
        focus = focus if focus in keys else keys[0]
        cam(focus)
        rest = [k for k in keys if k != focus]
        for col, k in zip(st.columns(len(rest)), rest):
            with col:
                cam(k, small=True)
        st.html(f'<div class="cf-foot">{E(DISCLOSURE)}. 상자는 미리 학습된 사람 검출 모델({E(eng.settings["detection"]["model"])})이 찾은 사람이고, 밝은 상자만 구역 안으로 셉니다.</div>')
    with col_map:
        st.html('<div class="cf-h">행사장 약도 · 요원 위치<span>가상 약도 · 축척 없음</span></div>')
        slots["map"] = st.empty()
        st.html('<div class="cf-legend"><span><i style="background:#F1F5F9"></i>CCTV가 찾은 사람</span><span><i style="background:#38BDF8"></i>근무</span>'
                '<span><i style="background:#FBBF24"></i>이동 중 · 곧 휴게</span><span><i style="background:#64748B"></i>휴게</span><span>요원의 자리는 구역 단위이고, 이동은 이동 시간 설정에 맞춘 그림입니다</span></div>')
    with col_side:
        slots["hero"] = st.empty()
        on_the_way = [s for s in eng.staff_view() if s["moving"]]
        if on_the_way:  # AI가 보낸 지시를 관리자가 멈출 수 있다 (D-58). 1초마다 새로 그리는 부분 밖에 둔다
            st.html('<div class="cf-h">진행 중인 지시<span>관리자가 취소할 수 있습니다</span></div>')
            for s in on_the_way:
                c1, c2 = st.columns([5, 1.2], vertical_alignment="center")
                c1.html(f'<div class="cf-line">{avatar_img(s["id"], 24)} <b>{E(s["name"])}</b> {E(place(eng, s["zone"]))} → {E(place(eng, s["target"]))} · {E(s["purpose"])}</div>')
                if c2.button("취소", key=f"cancel_{s['id']}", width="stretch"):
                    eng.submit(CANCEL, staff_id=s["id"])
                    st.toast(f"{s['name']}의 지시를 취소했습니다.")
        manager_form()
        st.html('<div class="cf-h">AI의 결정<span>줄을 누르면 토론의 핵심과 근거가 펼쳐집니다</span></div>')
        with st.container(height=270, border=False):
            slots["feed"] = st.empty()
        st.html('<div class="cf-h">요원 현황<span>모의 요원 · 그림은 실제 사람이 아닙니다</span></div>')
        slots["staff"] = st.empty()
        slots["raw"] = st.empty()
    manager_fast(slots)
    manager_live(slots)


# ---- 요원 화면 (휴대폰) --------------------------------------------------------------------------------
@st.fragment(run_every=UI["refresh_seconds"])
def staff_live(me: str, head) -> None:
    eng = engine()
    s = next(x for x in eng.staff_view() if x["id"] == me)
    if st.session_state.get("staff_sig") != (me, s["moving"], s["target"]):  # '도착' 단추가 생기거나 없어져야 하면 화면 전체를 다시 그린다
        st.rerun()
    ring = "#FBBF24" if s["moving"] else "#64748B" if s["state"] == "휴게" else "#38BDF8"
    where = f"{place(eng, s['zone'])} → {place(eng, s['target'])} 이동 중" if s["moving"] else f"{place(eng, s['zone'])} · {'휴게 중' if s['state'] == '휴게' else '근무 중'}"
    head.html(f'<div class="cf-me" style="--r:{ring}">{avatar_img(me, 58)}<div><h3>{E(s["name"])}</h3><p>{E(where)} · 다음 휴게 {s["next_break"]} · {eng.clock()}</p></div></div>')
    if eng.staff_script:  # 상황실에서 "요원 대본"을 켠 동안에는 이 화면에서 누르지 않아도 답과 도착이 들어간다 (D-60)
        st.html('<div class="cf-banner">지금은 요원 역할을 대본이 대신합니다. 답과 ‘도착’이 저절로 들어갈 수 있습니다.</div>')
    if s["state"] == "휴게":
        st.html(f'<div class="cf-banner">휴게 시간입니다 ({s["rest_left"]:.0f}분 남음). 휴게 중에는 지시가 오지 않습니다.</div>')
    msgs = [m for m in eng.room.all() if me in (m.to, m.sender) or m.to == "모두"]
    rows = []
    for m in msgs:
        mine = m.sender == me
        rows.append(f'<div class="cf-msg {"me" if mine else ""} {"order" if m.kind == chat.ORDER else ""}"><small>{E(m.clock)} · {"나" if mine else E(m.sender)} · {E(m.kind)}</small>{E(m.text)}</div>')
    said = {m.text for m in msgs[-3:] if m.sender == me}
    for trigger, payload in list(eng.waiting):
        if payload.get("staff_id") == me and payload.get("text", "도착") not in said:  # 이미 대화방에 올라간 글은 두 번 보이지 않게
            rows.append(f'<div class="cf-msg me" style="opacity:.6"><small>보냄 · 상황실이 읽는 중…</small>{E(payload.get("text", "도착"))}</div>')
    st.html('<div class="cf-msgs">' + ("".join(rows) or '<div class="cf-line" style="color:var(--muted)">아직 온 지시가 없습니다. 지시가 오면 여기에 보입니다.</div>') + "</div>")


def staff_page() -> None:
    eng = engine()
    st.html('<div class="cf-top"><div class="cf-brand">CrowdFlow <b>Agent</b><small>요원 화면</small></div></div>'
            '<div class="cf-foot" style="margin:0 0 6px">상황실의 지시를 받고, 글로 답하고, 도착하면 ‘도착’을 누릅니다. 녹화 영상으로 재현 · 요원은 모의 자료</div>')
    names = {s["id"]: s["name"] for s in eng.op.state.staff}
    ids = list(names)
    asked = st.query_params.get("staff")
    me = st.selectbox("나는 누구인가요?", ids, index=ids.index(asked) if asked in ids else None, format_func=lambda i: names[i], placeholder="이름을 고르세요")
    if me is None:
        st.info("이름을 고르면 나에게 온 지시가 보입니다.")
        return
    if asked != me:
        st.query_params["staff"] = me  # 주소에 남겨, 새로 고쳐도 같은 요원으로 들어오게 한다 (모의 요원 번호라 개인정보가 아님)
    s = next(x for x in eng.staff_view() if x["id"] == me)
    st.session_state["staff_sig"] = (me, s["moving"], s["target"])
    head = st.empty()
    if s["moving"]:  # '도착' 단추는 1초마다 새로 그리는 부분 밖에 둔다 (누르는 순간 다시 그려져 눌림이 사라지는 일이 없게)
        if st.button(f"도착 · {place(eng, s['target'])}에 왔습니다", type="primary", key=f"arrive_{me}", width="stretch"):
            eng.submit(ARRIVAL, staff_id=me)
            st.toast("도착을 알렸습니다.")
    staff_live(me, head)
    text = st.chat_input("답을 적어 주세요. 예: 갑니다 / 다친 관람객을 돌보는 중이라 못 갑니다")
    if text and text.strip():
        eng.submit(STAFF_TEXT, staff_id=me, text=text.strip())


# ---- 하루 정리 · 운영 지침 화면 (D-64) -----------------------------------------------------------------
def review_rows(it: dict) -> list[tuple[str, str]]:
    """정리 한 번을 견주기 표의 한 열로. 숫자는 모두 review.py가 실행 기록에서 센 것이다."""
    s = it["stats"]

    def n(v, unit: str) -> str:
        return "–" if v is None else f"{v}{unit}"

    script = "사람이 직접" if not it.get("staff_script") else f"{'어려운 날 대본' if it.get('hard_day') else '평소 대본'} (못 간다는 요원 {it.get('script_refusals', 1)}명)"
    return [
        ("운영 지침 읽기", f"켬 (조항 {it['clauses']}개)" if it["guideline_on"] else "끔"),
        ("요원 역할", script),
        ("붐빔 사건", f"{s['incidents_opened']}건 열림 · {s['incidents_closed']}건 닫음"),
        ("붐빔을 잡고 첫 지시까지 (평균)", n(s["first_order_wait_avg"], "초")),
        ("사건을 닫기까지 (평균)", n(s["seconds_to_close_avg"], "초")),
        ("보낼 요원이 없어 관리자에게 넘긴 일", f"{s['no_staff']}번"),
        ("휴게 교대 요원을 못 구한 일", f"{s['no_relief']}번"),
        ("최소 인원을 못 채운 일", f"{s['fill_short']}번"),
        ("지시받은 예비 요원", f"{s['reserves_used']} / {s['reserves_total']}명" + (f" ({s['reserves_all_used_clock']}에 다 씀)" if s.get("reserves_all_used_clock") else "")),
        ("토론에서 고른 선택지", " · ".join(f"{t} {c}번" for t, c in s.get("choices", {}).items()) or "–"),
        ("못 간다고 한 요원 · 다친 사람 보고", f"{s['refusals']}명 · {s['injuries']}건"),
        ("토론 (효율과 안전이 갈린 것)", f"{s['debates']}번 ({s['debates_split']}번)"),
        ("지침 조항이 해당한 토론 → 조항대로 정함", f"{s.get('guideline_applied', 0)}번 → {s.get('planner_followed_guideline', 0)}번" if it["guideline_on"] else "–"),
        ("요원 수 판단", it["review"]["staffing"]),
    ]


def days_table(items: list[dict]) -> str:
    cols = [review_rows(it) for it in items]
    head = "".join(f"<th>{it['day']}일차</th>" for it in items)
    rows = ""
    for i, (label, _) in enumerate(cols[0]):
        vals = [c[i][1] for c in cols]
        diff = len(set(vals)) > 1
        rows += f'<tr><td class="k">{E(label)}</td>' + "".join(f'<td class="{"diff" if diff else ""}">{E(v)}</td>' for v in vals) + "</tr>"
    return f'<table class="cf-opts"><tr><th></th>{head}</tr>{rows}</table>'


def review_html(it: dict) -> str:
    r = it["review"]

    def ul(items: list[str]) -> str:
        return '<ul class="cf-ul">' + "".join(f"<li>{E(x)}</li>" for x in items) + "</ul>"

    fixed = " (AI의 답이 숫자 기준과 달라 숫자대로 바로잡음)" if it.get("staffing_corrected") else ""
    return (f'<div class="cf-foot" style="margin-top:0">실행 기록에서 코드가 센 숫자</div><div class="cf-raw">{E(facts_text(it["stats"]))}</div>'
            f'<div class="cf-foot">아래 글은 위의 숫자만 보고 {E(it["wrote"])}이(가) 쓴 것입니다.</div>'
            f'<div class="cf-key" style="margin:10px 0 0">무엇이 문제였나</div>{ul(r["problems"])}'
            f'<div class="cf-key" style="margin:0">무엇이 더 필요했나</div>{ul(r["needs"])}'
            f'<div class="cf-key" style="margin:0">요원 수: {E(r["staffing"])}<span style="font-weight:500;color:var(--muted);font-size:12px">{fixed}</span></div>'
            f'<div class="cf-line" style="margin:4px 0 10px">{E(r["staffing_reason"])}</div>')


def debate_side(title: str, d: dict) -> str:
    body = ""
    for v in d["lines"]:
        cls, short = ROLES.get(v["who"], ("tool", v["who"][:2]))
        tail = "해당 조항 없음" if v["who"] == "지침 담당" and not v["option"] else f"선택지 {v['option']}"
        body += voice(cls, short, v.get("point") or "", tail, v["says"])
    return f'<div class="cf-card"><div class="cf-key">{E(title)} → 선택지 {d["chosen"]}</div>{body}</div>'


def review_page() -> None:
    eng = engine()
    running = eng.started and not eng.finished
    events = read_events(eng.log_path)
    clauses = eng.clauses()
    learned = [c for c in clauses if c.id.startswith("L-")]
    chips = [f'<span class="cf-chip {"ok" if eng.guideline_on else ""}"><i></i>지침 읽기 {"켬" if eng.guideline_on else "끔"} · 조항 {len(clauses)}개 (배운 것 {len(learned)}개)</span>',
             f'<span class="cf-chip {"warn" if eng.staff_script else ""}"><i></i>요원 역할: {"대본" + (" · 어려운 날" if eng.hard_day else "") if eng.staff_script else "사람이 직접"}</span>',
             f'<span class="cf-chip"><i></i>오늘 기록 {len(events)}건 · 영상 {eng.t:.0f} / {eng.total:.0f}초</span>']
    head, back = st.columns([8.4, 1.2], vertical_alignment="center")
    head.html(f'<div class="cf-top"><div class="cf-brand">CrowdFlow <b>Agent</b><small>하루 정리 · 운영 지침</small></div><div class="cf-chips">{"".join(chips)}</div></div>')
    back.page_link(PAGES["manager"], label="상황실로", icon=":material/arrow_back:", width="stretch")
    st.html('<div class="cf-foot" style="margin:0">녹화 영상으로 재현한 시연의 기록을 정리합니다. 숫자는 실행 기록에서 코드가 세고, 글은 그 숫자만 보고 확인 담당(AI)이 씁니다. '
            '운영 지침에는 관리자가 승인한 문장만 들어갑니다. 사람 수가 준 것은 영상이 원래 그런 것이지 요원을 보낸 결과가 아닙니다.</div>')

    refusals = eng.settings["scenario"]["staff_script"]["hard_refusals"]
    b1, b2, b3, b4 = st.columns([1.7, 1.5, 1.9, 1.9])
    done = eng.reviewed_gen == eng._gen
    if b1.button(f"{len(eng.reviews) + 1}일차 정리 쓰기", type="primary", disabled=running or done or not events, width="stretch",
                 help="영상이 끝난 뒤, 상황실에서 ‘처음부터 다시’를 누르기 전에 씁니다. 한 번 돌린 기록은 한 번만 정리합니다."):
        with st.spinner("기록에서 숫자를 세고, 확인 담당이 글을 쓰는 중…"):
            eng.make_review()
        st.rerun()
    if b2.button("지침 읽기 끄기" if eng.guideline_on else "지침 읽기 켜기", width="stretch",
                 help="켜면 토론 전에 운영 지침에서 지금 상황에 맞는 조항을 찾아 오고, 지침 담당이 해당하는 조항을 읽어 줍니다."):
        eng.guideline_on = not eng.guideline_on
        st.rerun()
    if b3.button("어려운 날 대본 끄기" if eng.hard_day else "어려운 날 대본 켜기", width="stretch",
                 help=f"켜면 요원 대본에서 붐빔 대응 지시를 처음 받은 요원 {refusals}명이 ‘다친 관람객을 돌보는 중이라 못 갑니다’라고 답합니다. "
                      "프로그램은 그대로 두고 현장 사정(대본)만 어렵게 한 것입니다. 켜면 요원 대본도 함께 켜집니다."):
        eng.hard_day = not eng.hard_day
        if eng.hard_day:
            eng.staff_script = True
        st.rerun()
    with b4.popover("정리·배운 지침 지우기", width="stretch"):
        st.write("지금까지 쓴 정리와, 승인해 더한 지침 문장을 모두 지웁니다. 기본 지침은 그대로 둡니다. 되돌릴 수 없습니다.")
        if st.button("지웁니다", key="forget"):
            eng.forget()
            st.rerun()
    if running:
        st.html('<div class="cf-banner">영상이 도는 중입니다. 영상이 끝나면 정리를 쓸 수 있습니다.</div>')
    elif done and eng.reviews:
        st.html(f'<div class="cf-banner">이번 실행은 {eng.reviews[-1]["day"]}일차로 정리했습니다. 다음 날을 돌리려면 상황실에서 ‘처음부터 다시’ → ‘시작’을 누르세요.</div>')

    left, right = st.columns([6.3, 3.7], gap="medium")
    with left:
        if len(eng.reviews) >= 2:
            st.html('<div class="cf-h">지난 날과 견주기<span>달라진 줄은 밝게 표시</span></div>' + days_table(eng.reviews[-3:])
                    + '<div class="cf-foot">같은 녹화 영상과 같은 대본으로 돌린 날끼리만 견줄 수 있습니다. AI의 답은 돌릴 때마다 조금씩 달라질 수 있어, 한 번의 비교로 '
                      '“지침 덕분에 나아졌다”고 단정하지 않습니다. ‘첫 지시까지’와 ‘닫기까지’는 AI가 생각한 시간이 들어간 값입니다.</div>')
        st.html('<div class="cf-h">하루 정리</div>')
        if not eng.reviews:
            st.html('<div class="cf-line" style="color:var(--muted)">아직 쓴 정리가 없습니다. 상황실에서 시연을 끝까지 돌린 뒤 ‘정리 쓰기’를 누르세요.</div>')
        have = {c.text for c in clauses}
        for it in reversed(eng.reviews):
            latest = it is eng.reviews[-1]
            with st.expander(f"**{it['day']}일차 정리** · 요원 수 ‘{it['review']['staffing']}’ · 지침 읽기 {'켬' if it['guideline_on'] else '끔'} · {it['made']}",
                             expanded=latest, icon=":material/summarize:", key=f"review-{it['day']}"):
                st.html(review_html(it))
                sug = it["review"]["suggestions"]
                st.html('<div class="cf-key" style="margin:0">지침에 더할 문장 (확인 담당의 제안)</div>'
                        + ('<div class="cf-line" style="color:var(--muted)">제안이 없습니다.</div>' if not sug else ""))
                if sug:
                    with st.form(f"approve-{it['day']}", border=False):
                        picks = [st.checkbox(s + ("  · 이미 지침에 있음" if s.strip() in have else ""), key=f"sug-{it['day']}-{i}", disabled=s.strip() in have)
                                 for i, s in enumerate(sug)]
                        if st.form_submit_button("고른 문장을 승인해 지침에 더하기", type="primary"):
                            n = eng.approve([s for s, p in zip(sug, picks) if p], f"{it['day']}일차 정리에서 제안 · 관리자 승인")
                            st.toast(f"지침에 문장 {n}개를 더했습니다." if n else "고른 문장이 없습니다.")
                            if n:
                                st.rerun()

        # 같은 상황, 같은 선택지를 지침 없이 한 번, 지침을 읽히고 한 번 토론시켜 견준다
        st.html('<div class="cf-h" style="margin-top:8px">같은 상황을 지침 없이 / 지침을 읽고 물어보기<span>달라지는 것은 지침을 읽었는지뿐</span></div>')
        plans = [(f"{it['day']}일차 {p['clock']} {p['zone']}구역 · {p['what']}", p) for it in eng.reviews for p in it.get("plans", [])]
        if not done:
            plans += [(f"오늘 {e.get('clock')} {e.get('zone')}구역 · {e['what']}", e) for e in events if e["kind"] == "계획" and e.get("situation")]
        if not plans:
            st.html('<div class="cf-line" style="color:var(--muted)">토론으로 정한 결정이 아직 없습니다. 시연을 돌리면 그때의 상황과 선택지가 여기에 올라옵니다.</div>')
        else:
            pick = st.selectbox("다시 물어볼 결정", range(len(plans)), index=len(plans) - 1, format_func=lambda i: plans[i][0], label_visibility="collapsed")
            p = plans[pick][1]
            with st.expander("이때의 상황과 선택지 (계산 도구가 만든 것 그대로)", key="cmp-situation"):
                st.html(f'<div class="cf-raw">{E(p["situation"])}\n\n{E(p["options_text"])}</div>')
            ai = not eng.brain.name.startswith("규칙")
            if st.button("지침 없이 한 번, 지침을 읽히고 한 번 물어보기", disabled=running or not ai or eng.model_ready is not True,
                         help="AI 담당들에게 같은 상황을 두 번 묻습니다. 40초쯤 걸립니다. 영상이 도는 동안에는 누를 수 없습니다."):
                try:
                    with st.spinner("같은 상황을 지침 없이 한 번, 지침을 읽히고 한 번 토론시키는 중… (40초쯤)"):
                        eng.compare(p["situation"], p["options_text"], p["query"])
                    st.rerun()
                except Exception as e:  # 임베딩 모델이나 AI 서버에 닿지 않음 등
                    st.error(f"물어보지 못했습니다: {type(e).__name__}: {str(e)[:200]}")
        for i, c in enumerate(reversed(eng.compares[-3:])):
            same = c["without"]["chosen"] == c["with"]["chosen"]
            title = (f"**지침 없이 선택지 {c['without']['chosen']} → 지침을 읽고 선택지 {c['with']['chosen']}**"
                     + (" (같은 답)" if same else " (답이 달라짐)") + f" · {c['made']}")
            with st.expander(title, expanded=i == 0, icon=":material/compare_arrows:", key=f"cmp-{c['made']}"):
                found = c["found"]
                rows = "".join(f'<span><b>{E(x["id"])}</b> {E(x["text"])}' + (f' · 가까운 정도 {x["score"]:.2f}' if x.get("score") is not None else "") + "</span>"
                               for x in found["clauses"])
                st.html(f'<div class="cf-foot" style="margin-top:0">지침에서 찾아 읽은 조항 · {E(found["mode"])} (조항 {found["total"]}개 가운데 {len(found["clauses"])}개)</div>'
                        f'<div class="cf-ex">{rows}</div>')
                c1, c2 = st.columns(2)
                c1.html(debate_side("지침 없이", c["without"]))
                c2.html(debate_side("지침을 읽고", c["with"]))
                st.html('<div class="cf-foot">같은 상황과 같은 선택지를 AI 담당들에게 두 번 물은 것입니다. AI의 답은 물을 때마다 조금씩 달라질 수 있습니다.</div>')
    with right:
        st.html(f'<div class="cf-h">운영 지침<span>기본 {len(clauses) - len(learned)}개 · 승인해 더한 것 {len(learned)}개</span></div>')
        rows = "".join(f'<div class="cf-cl {"new" if c.id.startswith("L-") else ""}"><b>{E(c.id)}</b><div>{E(c.text)}<small>{E(c.source)}</small></div></div>' for c in clauses)
        st.html(f'<div class="cf-card" style="padding:6px 14px">{rows}</div>'
                '<div class="cf-foot">기본 지침은 이미 정한 운영 규칙을 옮겨 적은 시연용 모의 자료입니다. 조항이 '
                f'{eng.guide_cfg["top_k"]}개보다 많으면 토론 때 지금 상황에 가장 가까운 {eng.guide_cfg["top_k"]}개만 찾아 읽습니다.</div>')
        with st.form("own-clause", clear_on_submit=True, border=False):
            f1, f2 = st.columns([5, 1.3], vertical_alignment="bottom")
            own = f1.text_input("관리자가 직접 쓰는 문장", placeholder="관리자가 직접 지침 문장 쓰기", label_visibility="collapsed")
            if f2.form_submit_button("더하기", width="stretch") and own.strip():
                eng.approve([own.strip()], "관리자가 직접 씀")
                st.rerun()


# ---- 결과 보고서 화면 (D-75) ---------------------------------------------------------------------------
STAFFING_COLOR = {"모자람": "var(--bad)", "적당함": "var(--ok)", "넉넉함": "var(--accent)"}


def report_md(it: dict) -> str:
    return result_report_markdown(it["stats"], it["review"], it["made"], it["wrote"], bool(it.get("staffing_corrected")))


def report_html(it: dict) -> str:
    """결과 보고서 한 장. 숫자와 표는 review.py가 실행 기록에서 센 것이고, 오른쪽의 정리 글만 확인 담당(AI)이 쓴 것이다."""
    s, r = it["stats"], it["review"]

    def ul(items: list[str]) -> str:
        return '<ul class="cf-ul">' + "".join(f"<li>{E(x)}</li>" for x in items) + "</ul>"

    kpis = "".join(f'<div class="cf-kpi"><div class="l">{E(n)}</div><div class="v">{E(v)}</div><div class="s">{E(note)}</div></div>' for n, v, note in headline_numbers(s))
    rows = "".join(f'<tr><td><b>{E(x["zone"])}</b></td><td>{E(x["opened"])}</td><td>{E(x["wait"])}</td><td>{E(x["sent"])}</td><td>{E(x["rechecks"])}</td>'
                   f'<td>{E(x["handed"])}</td><td style="color:{"var(--ok)" if x["closed"] else "var(--warn)"}">{E(x["result"])}</td></tr>' for x in incident_rows(s))
    table = ('<table class="cf-opts"><tr><th>구역</th><th>시작 · 단계</th><th>첫 지시까지</th><th>지시받은 요원</th><th>다시 세기</th><th>관리자에게 넘김</th><th>결과</th></tr>'
             + (rows or '<tr><td colspan="7">붐빔 사건이 없었습니다.</td></tr>') + "</table>")
    fixed = " · AI의 답이 숫자 기준과 달라 숫자대로 바로잡음" if it.get("staffing_corrected") else ""
    return (f'<div class="cf-rep"><div class="ttl">행사 안전 운영 결과 보고서</div>'
            f'<div class="meta">작성 {E(it["made"])} · 실행 기록 {s["events"]}건 (마지막 기록은 영상 {s["video_seconds"]}초) · '
            f'숫자와 표: 코드가 실행 기록에서 셈 · 정리 글: {E(writer_label(it["wrote"]))}</div>'
            f'<div class="sec">1. 한눈에 보기</div><div class="cf-kpis">{kpis}</div>'
            f'<div class="cols"><div><div class="sec">2. 사건별 경과</div>{table}'
            f'<div class="sec">3. 현장에서 들어온 일과 운영 조건</div>{ul(field_notes(s))}</div>'
            f'<div><div class="sec">4. 확인 담당의 정리 (AI가 위의 숫자만 보고 쓴 글)</div>'
            f'<div class="cf-key" style="margin:6px 0 0">무엇이 문제였나</div>{ul(r["problems"])}'
            f'<div class="cf-key" style="margin:0">무엇이 더 필요했나</div>{ul(r["needs"])}'
            f'<div class="cf-key" style="margin:0">요원 수<span class="verdict" style="color:{STAFFING_COLOR.get(r["staffing"], "var(--text)")}">{E(r["staffing"])}</span></div>'
            f'<div class="cf-line" style="margin:5px 0 0">{E(r["staffing_reason"])}<span style="color:var(--muted)">{E(fixed)}</span></div></div></div>'
            f'<div class="cf-foot">{E(REPORT_NOTICE)}</div></div>')


def report_page() -> None:
    """영상이 끝난 뒤 그 실행의 결과 보고서를 쓰고 보여 준다. 운영 지침(RAG)에 넣을 제안과 견주기는 여기에 없다 (D-68, D-75)."""
    eng = engine()
    running = eng.started and not eng.finished
    events = read_events(eng.log_path)
    done = eng.reviewed_gen == eng._gen
    current = eng.reviews[-1] if done and eng.reviews else None
    chips = [f'<span class="cf-chip"><i></i>이번 실행의 기록 {len(events)}건 · 영상 {eng.t:.0f} / {eng.total:.0f}초</span>',
             '<span class="cf-chip"><i></i>녹화 영상 재현</span>']
    head, back = st.columns([8.4, 1.2], vertical_alignment="center")
    head.html(f'<div class="cf-top"><div class="cf-brand">CrowdFlow <b>Agent</b><small>결과 보고서</small></div><div class="cf-chips">{"".join(chips)}</div></div>')
    back.page_link(PAGES["manager"], label="상황실로", icon=":material/arrow_back:", width="stretch")
    b1, b2, note = st.columns([1.6, 1.6, 6.4], vertical_alignment="center")
    if b1.button("결과 보고서 쓰기", type="primary", disabled=running or done or not events, width="stretch",
                 help="영상이 끝난 뒤, 상황실에서 ‘처음부터 다시’를 누르기 전에 씁니다. 숫자와 표는 코드가 실행 기록에서 세고, 정리 글은 확인 담당(AI)이 그 숫자를 보고 씁니다."):
        with st.spinner("실행 기록에서 숫자를 세고, 확인 담당이 정리하는 중…"):
            item = eng.make_review()
            stamp = item["made"].replace("-", "").replace(":", "").replace(" ", "_")
            (eng.reports_dir / f"report_{stamp}.md").write_text(report_md(item), encoding="utf-8")  # 보고서 꼴의 글도 파일로 남긴다
        st.rerun()
    if current:
        b2.download_button("보고서 파일 받기", data=report_md(current), file_name="crowdflow_result_report.md", mime="text/markdown", width="stretch",
                           icon=":material/download:")
    msg = ("영상이 도는 중입니다. 영상이 끝나면 결과 보고서를 쓸 수 있습니다." if running else
           "이번 실행의 결과 보고서입니다. 다시 돌리려면 상황실에서 ‘처음부터 다시’ → ‘시작’을 누르세요." if current else
           "상황실에서 시연을 끝까지 돌린 뒤 ‘결과 보고서 쓰기’를 누르세요." if not events else "‘결과 보고서 쓰기’를 누르면 이번 실행의 기록으로 보고서를 씁니다.")
    note.html(f'<div class="cf-line" style="color:var(--muted)">{E(msg)}</div>')
    if current:
        st.html(report_html(current))
    past = eng.reviews[:-1] if current else eng.reviews
    if past:
        with st.expander(f"지난 결과 보고서 {len(past)}건", icon=":material/history:", key="past-reports"):
            for it in reversed(past):
                st.html(report_html(it))


st.set_page_config(page_title="CrowdFlow Agent", layout="wide")
st.html(f"<style>{CSS}{MAP_CSS}</style>")
PAGES = {"manager": st.Page(manager_page, title="상황실", default=True), "staff": st.Page(staff_page, title="요원 화면", url_path="staff")}
if UI.get("review_screen"):  # 기본은 감춤 (D-68). 설정을 켜면 하루 정리 · 운영 지침 화면 전체와 관리자 화면의 단추가 생긴다
    PAGES["review"] = st.Page(review_page, title="하루 정리", url_path="review")
elif UI.get("result_report"):  # 결과 보고서만 보여 준다 (D-75). 운영 지침(RAG) 쪽은 감춘 그대로다
    PAGES["review"] = st.Page(report_page, title="결과 보고서", url_path="report")
st.navigation(list(PAGES.values()), position="hidden").run()
