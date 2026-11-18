# 이 파일이 하는 일: 관리자 화면에 넣는 행사장 약도(SVG 그림)와 요원 그림(아바타)을 만든다.
#   약도에는 길, 구역별 붐빔 색, CCTV가 찾은 사람 점, CCTV가 보는 방향, 요원의 자리와 이동이 나온다.
#   요원의 자리는 구역 단위로만 안다(D-10). 그래서 이동하는 모습은 "출발 구역 → 도착 구역"을 이동 시간에 맞춰 그린 그림이고,
#   실제 위치를 잰 것이 아니다. 요원 그림도 실제 사람의 사진이 아니라 모의 요원을 구별하려고 그린 그림이다.
# 해당 기능: 핵심 기능 1 (사람 세기)과 3·4 (요원 배치, 실행)를 보여 주는 부분
# 해당 기준: C-16, C-31, C-42 (개인정보: 실제 사진을 쓰지 않음), C-43
# 근거: D-13 (관리자 화면: 구역별 붐빔 색, 요원 위치와 상태), D-10 (위치는 구역 단위), D-46 (약도의 모양: 길로 이어진 떨어진 구역),
#   D-55 (화면을 다시 만듦: 약도를 새로 그리고 요원 그림과 이동을 넣음. 사용자 지시 2026-10-04), 01의 11번 (관측 불가 표시),
#   D-61 (예비 요원이 5명이 되어 대기소를 두 줄로 그림)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import base64
import functools
import html

from crowdflow.venue import Venue

S = 40.0  # 약도 단위 하나를 몇 픽셀로 그릴지 (화면 설정. 계산에는 쓰지 않음)
PAD = 14.0
LANE_GAP = 0.65  # 요원 그림을 세우는 줄: 길의 왼쪽 끝에서 이만큼 안쪽 (약도 단위)
SLOT_STEP = 1.26  # 요원 그림 사이의 세로 간격 (약도 단위)

# 요원마다 다른 그림: 배경색, 살색, 머리색, 머리 모양. 예비 요원은 조끼 색이 다르다
AVATARS = {
    "S1": ("#2563EB", "#F1C9A8", "#2A211D", "short"), "S2": ("#7C3AED", "#E9B894", "#1F1A17", "bob"),
    "S3": ("#0E9AA7", "#F4D0B0", "#3A2A20", "cap"), "S4": ("#DB2777", "#EBC09E", "#1C1614", "bun"),
    "S5": ("#EA580C", "#F1C9A8", "#2B2320", "side"), "S6": ("#16A34A", "#DFA982", "#15110F", "buzz"),
    "S7": ("#CA8A04", "#F4D0B0", "#2A211D", "bob"), "S8": ("#4F46E5", "#E9B894", "#33261E", "short"),
    "R1": ("#64748B", "#F1C9A8", "#2A211D", "cap"), "R2": ("#57534E", "#E9B894", "#1F1A17", "short"),
    "R3": ("#52525B", "#F4D0B0", "#3A2A20", "bob"), "R4": ("#475569", "#DFA982", "#15110F", "buzz"),
    "R5": ("#6B7280", "#EBC09E", "#2B2320", "side"),
}
POST_COLS = 3  # 대기소 한 줄에 세우는 예비 요원 수 (화면 설정)
HAIR = {
    "short": '<path d="M21 26.5Q20.3 15 32 15Q43.7 15 43 26.5Q40 21.6 32 21.6Q24 21.6 21 26.5Z"/>',
    "side": '<path d="M20.8 27Q20 14.6 32 14.6Q44 14.6 43.2 27L41.6 23.8Q34 23 29.4 19.6Q26.4 23.4 20.8 27Z"/>',
    "bob": '<path d="M20.6 37Q18.4 14.4 32 14.4Q45.6 14.4 43.4 37L40.8 37Q42.6 24 38 21.4Q32 19.8 26 21.4Q21.4 24 23.2 37Z"/>',
    "bun": '<circle cx="32" cy="13.2" r="4.2"/><path d="M21 26.5Q20.3 15 32 15Q43.7 15 43 26.5Q40 21.2 32 21.2Q24 21.2 21 26.5Z"/>',
    "buzz": '<path d="M21.4 25Q21.6 15.6 32 15.6Q42.4 15.6 42.6 25Q38 19.8 32 19.8Q26 19.8 21.4 25Z"/>',
    "cap": '<path d="M20.6 24.6Q21 14.4 32 14.4Q43 14.4 43.4 24.6Z" fill="#E2E8F0"/><path d="M19 24.2H45Q46.5 24.2 46.5 25.6Q46.5 27 45 27H21.5Q19 27 19 24.2Z" fill="#CBD5E1"/>',
}


def avatar_inner(staff_id: str) -> str:
    """요원 그림의 속 (가로세로 64짜리 틀). 얼굴을 알아볼 수 있는 사진이 아니라 단순한 그림이다."""
    bg, skin, hair, style = AVATARS.get(staff_id, ("#475569", "#F1C9A8", "#2A211D", "short"))
    vest = "#FB923C" if staff_id.startswith("R") else "#D9F53F"
    return (f'<circle cx="32" cy="32" r="32" fill="{bg}"/>'
            '<path d="M9 64Q9 45 32 45Q55 45 55 64Z" fill="#1E293B"/>'  # 어깨 (근무복)
            f'<path d="M17.5 64L22 47.2L32 57L42 47.2L46.5 64Z" fill="{vest}"/>'  # 안전 조끼
            '<path d="M19.4 57.2H44.6" stroke="#F8FAFC" stroke-width="1.8" opacity=".85"/>'  # 반사 띠
            f'<rect x="28.2" y="34.5" width="7.6" height="9" rx="3.2" fill="{skin}"/>'
            f'<circle cx="32" cy="27" r="10.6" fill="{skin}"/>'
            f'<g fill="{hair}">{HAIR[style]}</g>'
            '<circle cx="28.3" cy="28.6" r="1.05" fill="#1F2937"/><circle cx="35.7" cy="28.6" r="1.05" fill="#1F2937"/>'
            '<path d="M29.2 32.4Q32 34.4 34.8 32.4" stroke="#1F2937" stroke-width="1" fill="none" stroke-linecap="round"/>')


def svg_img(svg: str, cls: str, extra: str = "") -> str:
    """SVG 그림을 <img>로 감싼다. Streamlit의 st.html이 SVG 태그를 지우기 때문에(E-53) 그림 파일처럼 넣는다."""
    return f'<img class="{cls}" {extra} alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode("utf-8")).decode("ascii")}">'


@functools.lru_cache(maxsize=64)
def avatar_img(staff_id: str, size: int = 40) -> str:
    """HTML 안에 넣는 요원 그림."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 64 64">'
           f'<defs><clipPath id="c"><circle cx="32" cy="32" r="32"/></clipPath></defs><g clip-path="url(#c)">{avatar_inner(staff_id)}</g></svg>')
    return svg_img(svg, "av", f'width="{size}" height="{size}"')


def _lane_x(venue: Venue) -> float:
    road_left = min((r[0] for r in venue.roads if r[3] > r[2]), default=0.0)  # 세로 길의 왼쪽 끝
    return road_left + LANE_GAP


def staff_positions(venue: Venue, staff: list[dict], standby: str) -> dict[str, dict]:
    """요원마다 약도 위의 자리를 정한다. staff의 항목: id, zone, moving(bool), target, progress(0~1), progress_prev.

    구역에 있는 요원은 구역 왼쪽의 줄에 번호 차례로 세우고, 이동 중인 요원은 출발 자리와 도착 자리 사이에 놓는다.
    돌려주는 값: id → {"xy": (u, v), "prev": (u, v), "dest": (u, v) 또는 None}
    """
    lane = _lane_x(venue)
    post = _post_rect(venue, _post_rows(staff))

    lane_right = max(r[0] + r[2] for r in venue.roads if r[3] > r[2]) - LANE_GAP  # 길의 오른쪽 줄 (왼쪽 줄이 차면 여기에 세운다)

    def slot(zone: str, i: int) -> tuple[float, float]:
        if zone in venue.zones:
            _, v0, _, h = venue.zones[zone].plan_rect
            col, row = divmod(i, max(1, int((h - 0.5) // SLOT_STEP) + 1))
            return (lane if col % 2 == 0 else lane_right), v0 + 0.6 + row * SLOT_STEP
        row, col = divmod(i, POST_COLS)  # 대기소 안: 한 줄에 POST_COLS명씩
        return post[0] + 0.75 + col * 1.15, post[1] + 0.97 + row * SLOT_STEP

    order = lambda s: (s["id"].startswith("R"), s["id"])
    at: dict[str, list[dict]] = {}
    for s in sorted(staff, key=order):
        at.setdefault(s["zone"], []).append(s)
    out: dict[str, dict] = {}
    arriving: dict[str, int] = {}
    for zone, people in at.items():
        for i, s in enumerate(people):
            home = slot(zone, i)
            if not s.get("moving"):
                out[s["id"]] = {"xy": home, "prev": home, "dest": None}
                continue
            k = arriving.get(s["target"], 0)
            arriving[s["target"]] = k + 1
            dest = slot(s["target"], len(at.get(s["target"], [])) + k)
            # 요원 그림은 움직이지 않는다(실제 위치를 모르므로). 출발 자리에 그대로 두고, 가는 곳까지 점선만 긋는다
            out[s["id"]] = {"xy": home, "prev": home, "dest": dest}
    return out


def _along(path: list[tuple[float, float]], p: float) -> tuple[float, float]:
    """꺾인 선(path) 위에서 전체 길이의 p(0~1)만큼 간 점."""
    seg = [((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 for (x0, y0), (x1, y1) in zip(path, path[1:])]
    total, left = sum(seg) or 1.0, max(0.0, min(1.0, p)) * (sum(seg) or 1.0)
    for ((x0, y0), (x1, y1)), d in zip(zip(path, path[1:]), seg):
        if left <= d or d == 0:
            f = 0.0 if d == 0 else left / d
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        left -= d
    return path[-1]


def _post_rows(staff: list[dict]) -> int:
    """대기소에 몇 줄이 필요한지. 예비 요원의 수로 정한다(요원이 나가도 크기가 바뀌지 않게, 지금 있는 수가 아니라 전체 수로)."""
    reserves = sum(1 for s in staff if s.get("reserve"))
    return max(1, -(-reserves // POST_COLS))


def _post_above(venue: Venue) -> int:
    """대기소 바로 위에 있는 구역이 몇 번째인지 (가운데 두 구역의 사이에 대기소를 둔다)."""
    return max(len(venue.zones) // 2 - 1, 0)


def _post_rect(venue: Venue, rows: int = 1) -> tuple[float, float, float, float]:
    """예비 요원이 기다리는 곳(대기소)의 네모 [왼쪽, 위, 가로, 세로]. 가운데 두 구역의 사이, 길의 왼쪽에 둔다."""
    rects = [z.plan_rect for z in venue.zones.values()]
    i = _post_above(venue)
    gap_y = (rects[i][1] + rects[i][3] + rects[i + 1][1]) / 2 if len(rects) > 1 else rects[0][1]
    road_left = _lane_x(venue) - LANE_GAP
    w = 0.75 * 2 + (POST_COLS - 1) * 1.15
    h = 1.7 + (rows - 1) * SLOT_STEP
    return road_left - 0.4 - w, gap_y - h / 2, w, h


def render_map(venue: Venue, status: dict[str, dict], points: dict, staff: list[dict], colors: list[str], standby: str,
               t: float = 0.0, note: str = "") -> str:
    """약도 SVG 한 장.

    status: 구역 → {"count", "count_avg", "level_index", "level_name"}
    points: 구역 → 약도 좌표 (N×2). CCTV가 찾은 사람의 발 위치
    staff: 요원 목록. 항목: id, name, zone, state("근무"|"이동"|"휴게"), soon(bool), blocked(bool), moving, target, progress, progress_prev, fresh(bool)
    colors: 붐빔 단계의 색 (평상, 주의, 경계, 심각)
    t: 영상의 몇 초인지 (움직이는 무늬가 끊기지 않게 이어 그리는 데 쓴다)
    """
    pw, ph = venue.plan_size
    W, H = pw * S + 2 * PAD, ph * S + 2 * PAD
    X = lambda u: PAD + float(u) * S
    Y = lambda v: PAD + float(v) * S
    lane = _lane_x(venue)
    road_left = lane - LANE_GAP
    post_rows = _post_rows(staff)
    o: list[str] = []
    o.append(f'<svg viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" xmlns="http://www.w3.org/2000/svg">')
    o.append(f'<style>{MAP_SVG_CSS}</style><defs>'
             '<pattern id="cf-hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="9" height="9" fill="#141C30"/><line x1="0" y1="0" x2="0" y2="9" stroke="#24304D" stroke-width="3"/></pattern>'
             '<pattern id="cf-dots" width="16" height="16" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#1B2540"/></pattern>'
             '<linearGradient id="cf-cone" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#FFFFFF" stop-opacity=".10"/><stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>'
             '<filter id="cf-glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
             + "".join(f'<clipPath id="cf-av-{s["id"]}"><circle cx="32" cy="32" r="32"/></clipPath>' for s in staff) + '</defs>')
    o.append(f'<rect width="{W:.0f}" height="{H:.0f}" rx="14" fill="#0E1526"/><rect width="{W:.0f}" height="{H:.0f}" rx="14" fill="url(#cf-dots)"/>')

    # 길: 먼저 바탕(가장자리 선)을 그리고 속을 채워, 길끼리 만나는 곳에 선이 남지 않게 한다
    for u0, v0, w, h in venue.roads:
        o.append(f'<rect x="{X(u0) - 1.5:.1f}" y="{Y(v0) - 1.5:.1f}" width="{w * S + 3:.1f}" height="{h * S + 3:.1f}" rx="10" fill="#2A385C"/>')
    for u0, v0, w, h in venue.roads:
        o.append(f'<rect x="{X(u0):.1f}" y="{Y(v0):.1f}" width="{w * S:.1f}" height="{h * S:.1f}" rx="9" fill="#18223B"/>')
    for u0, v0, w, h in venue.roads:  # CCTV가 없는 길(가로로 난 길)은 빗금으로 "관측 불가"를 알린다 (01의 11번)
        if w > h:
            o.append(f'<rect x="{X(u0) + 6:.1f}" y="{Y(v0) + 6:.1f}" width="{(road_left - u0) * S - 12:.1f}" height="{h * S - 12:.1f}" rx="6" fill="url(#cf-hatch)" opacity=".9"/>')
            o.append(f'<text x="{X(u0 + (road_left - u0) / 2):.1f}" y="{Y(v0 + h / 2) + 4:.1f}" class="cf-muted" text-anchor="middle">CCTV 없음 · 관측 불가</text>')

    zones = list(venue.zones.items())
    for i, (k, z) in enumerate(zones):
        s = status[k]
        c = colors[min(s["level_index"], len(colors) - 1)]
        u0, v0, w, h = z.plan_rect
        hot = s["level_index"] > 0
        # CCTV가 보는 방향: 가까운 쪽(아래)에서 먼 쪽(위)을 본다
        cx = X(u0 + w / 2)
        o.append(f'<polygon points="{cx - 9:.1f},{Y(v0 + h):.1f} {cx + 9:.1f},{Y(v0 + h):.1f} {X(u0 + w):.1f},{Y(v0):.1f} {X(u0):.1f},{Y(v0):.1f}" fill="url(#cf-cone)"/>')
        pulse = f' filter="url(#cf-glow)" style="animation:cf-pulse 1.6s ease-in-out infinite;animation-delay:-{t % 1.6:.2f}s"' if hot else ""
        o.append(f'<rect x="{X(u0):.1f}" y="{Y(v0):.1f}" width="{w * S:.1f}" height="{h * S:.1f}" rx="10" fill="{c}" fill-opacity="{0.20 if hot else 0.10}" '
                 f'stroke="{c}" stroke-width="{2.4 if hot else 1.4}"{pulse}/>')
        pts = points.get(k)
        if pts is not None:
            for u, v in pts:
                o.append(f'<circle cx="{X(u):.1f}" cy="{Y(v):.1f}" r="3.3" fill="#F1F5F9" stroke="#0B1120" stroke-opacity=".55" stroke-width="1"/>')
        # 구역 이름표와 CCTV 표시
        o.append(f'<rect x="{X(u0) + 8:.1f}" y="{Y(v0) + 8:.1f}" width="26" height="22" rx="7" fill="{c}"/><text x="{X(u0) + 21:.1f}" y="{Y(v0) + 24:.1f}" class="cf-tag" text-anchor="middle">{html.escape(k)}</text>')
        o.append(f'<g transform="translate({cx - 11:.1f},{Y(v0 + h) - 9:.1f})"><rect width="22" height="14" rx="4" fill="#0B1120" stroke="#64748B"/><circle cx="11" cy="7" r="3.4" fill="#94A3B8"/><circle cx="11" cy="7" r="1.5" fill="#0B1120"/></g>')
        # 길 왼쪽의 안내: 구역 이름, 거리, 사람 수, 단계. 대기소와 겹치지 않게 구역의 가운데 높이에 둔다
        ly = Y(v0 + h / 2) if i < len(zones) - 1 else Y(v0 + 1.25)
        if post_rows > 1 and i == _post_above(venue):  # 대기소가 두 줄 이상이면 바로 위 구역의 안내를 위로, 바로 아래 구역의 안내를 아래로 비킨다
            ly -= 0.75 * S
        elif post_rows > 1 and i == _post_above(venue) + 1 and i < len(zones) - 1:
            ly += 0.3 * S
        lx = X(road_left) - 16
        o.append(f'<text x="{lx:.1f}" y="{ly - 20:.1f}" class="cf-zone" text-anchor="end">{html.escape(z.name)}</text>'
                 f'<text x="{lx:.1f}" y="{ly - 3:.1f}" class="cf-muted" text-anchor="end">{html.escape(z.street)} · {html.escape(z.cctv.id)}</text>'
                 f'<text x="{lx:.1f}" y="{ly + 28:.1f}" class="cf-num" text-anchor="end" fill="{c}">{s["count_avg"]:.0f}<tspan class="cf-unit" dx="3">명</tspan></text>'
                 f'<rect x="{lx - 46:.1f}" y="{ly + 37:.1f}" width="46" height="19" rx="9.5" fill="{c}" fill-opacity=".16" stroke="{c}" stroke-opacity=".7"/>'
                 f'<text x="{lx - 23:.1f}" y="{ly + 50.5:.1f}" class="cf-pill" text-anchor="middle" fill="{c}">{html.escape(s["level_name"])}</text>')

    # 대기소 (예비 요원이 기다리는 곳)
    px, py, pww, phh = _post_rect(venue, post_rows)
    o.append(f'<path d="M{X(px + pww):.1f} {Y(py + phh / 2):.1f}H{X(lane):.1f}" stroke="#2A385C" stroke-width="12" stroke-linecap="round"/>'
             f'<rect x="{X(px):.1f}" y="{Y(py):.1f}" width="{pww * S:.1f}" height="{phh * S:.1f}" rx="10" fill="#131C31" stroke="#2A385C" stroke-dasharray="5 4"/>'
             f'<text x="{X(px) + 10:.1f}" y="{Y(py) + 16:.1f}" class="cf-muted">{html.escape(standby)}소 · 예비 요원</text>')

    # 요원: 이동 중인 길 → 그림
    pos = staff_positions(venue, staff, standby)
    ring = {"근무": "#38BDF8", "이동": "#FBBF24", "휴게": "#64748B"}
    for s in staff:
        p = pos[s["id"]]
        if p["dest"]:
            side = X(lane) - 27  # 다른 요원 그림을 가리지 않게, 줄의 왼쪽으로 비켜서 긋는다
            o.append(f'<path d="M{X(p["xy"][0]) - 19:.1f} {Y(p["xy"][1]):.1f}H{min(side, X(p["xy"][0]) - 27):.1f}V{Y(p["dest"][1]):.1f}H{X(p["dest"][0]) - 19:.1f}" fill="none" stroke="#FBBF24" stroke-width="2" stroke-dasharray="5 6" '
                     f'style="animation:cf-march .8s linear infinite;animation-delay:-{t % 0.8:.2f}s" opacity=".9"/>'
                     f'<circle cx="{X(p["dest"][0]):.1f}" cy="{Y(p["dest"][1]):.1f}" r="17" fill="none" stroke="#FBBF24" stroke-dasharray="3 4" opacity=".7"/>')
    for s in sorted(staff, key=lambda s: s["state"] == "이동"):  # 이동 중인 요원을 맨 위에 그린다
        p = pos[s["id"]]
        x, y, x0, y0 = X(p["xy"][0]), Y(p["xy"][1]), X(p["prev"][0]), Y(p["prev"][1])
        col = ring[s["state"]]
        move = (f' style="--dx:{x0 - x:.1f}px;--dy:{y0 - y:.1f}px;animation:cf-move 1s linear both"' if (x0, y0) != (x, y)
                else ' style="animation:cf-pop .7s ease-out both"' if s.get("fresh") else "")
        dim = ' opacity=".4"' if s["state"] == "휴게" else ""
        dash = ' stroke-dasharray="4 3"' if s["state"] == "이동" else ""
        o.append(f'<g transform="translate({x:.1f},{y:.1f})"><g{move}>')
        o.append(f'<circle r="18.5" fill="#0B1120"/><g transform="translate(-16,-16) scale(.5)" clip-path="url(#cf-av-{s["id"]})"{dim}>{avatar_inner(s["id"])}</g>'
                 f'<circle r="17.2" fill="none" stroke="{col}" stroke-width="2.6"{dash}/>')
        badge = ("휴게", "#475569", "#E2E8F0") if s["state"] == "휴게" else ("이동 중", "#92400E", "#FDE68A") if s["state"] == "이동" else \
                ("못 감", "#B91C1C", "#FEE2E2") if s.get("blocked") else \
                ("곧 휴게", "#92400E", "#FDE68A") if s.get("soon") else None
        if badge:
            bw = 12 + 10 * len(badge[0])
            o.append(f'<rect x="{9:.0f}" y="-25" width="{bw}" height="15" rx="7.5" fill="{badge[1]}"/><text x="{9 + bw / 2:.1f}" y="-14.2" class="cf-badge" text-anchor="middle" fill="{badge[2]}">{badge[0]}</text>')
        nw = 10 + 9.5 * len(s["name"])
        o.append(f'<rect x="{-nw / 2:.1f}" y="12" width="{nw:.1f}" height="14" rx="7" fill="#0B1120" stroke="{col}" stroke-opacity=".8"/>'
                 f'<text y="22.3" class="cf-name" text-anchor="middle">{html.escape(s["name"])}</text></g></g>')
    if note:
        o.append(f'<text x="{PAD + 6:.0f}" y="{PAD + 16:.0f}" class="cf-muted">{html.escape(note)}</text>')
    o.append("</svg>")
    return "".join(o)


MAP_CSS = ".cf-map{display:block;margin:0 auto;width:auto;height:auto;max-width:100%;max-height:calc(100vh - 150px);border-radius:14px}"  # 화면 쪽: 약도 그림의 크기

# 약도 그림 안에서 쓰는 글꼴과 움직임. 그림 안에서는 화면의 웹 글꼴을 쓸 수 없어, 컴퓨터에 있는 글꼴로 그린다
MAP_SVG_CSS = """
svg{font-family:'Pretendard Variable','Pretendard','Malgun Gothic','Apple SD Gothic Neo',sans-serif}
.cf-zone{font-size:15px;font-weight:700;fill:#E6ECF7}
.cf-muted{font-size:10.5px;fill:#7C8BA8}
.cf-num{font-size:27px;font-weight:700;font-family:Bahnschrift,'Malgun Gothic',sans-serif}
.cf-unit{font-size:11px;font-weight:400;fill:#7C8BA8;font-family:'Malgun Gothic',sans-serif}
.cf-pill{font-size:11px;font-weight:700}
.cf-tag{font-size:13px;font-weight:800;fill:#0B1120}
.cf-name{font-size:9.5px;font-weight:600;fill:#E6ECF7}
.cf-badge{font-size:9px;font-weight:700}
@keyframes cf-pulse{0%,100%{stroke-opacity:1;fill-opacity:.22}50%{stroke-opacity:.45;fill-opacity:.10}}
@keyframes cf-march{to{stroke-dashoffset:-11}}
@keyframes cf-move{from{transform:translate(var(--dx),var(--dy))}to{transform:translate(0,0)}}
@keyframes cf-pop{0%{transform:scale(1.45)}100%{transform:scale(1)}}
"""
