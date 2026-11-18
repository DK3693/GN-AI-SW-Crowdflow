# 이 파일이 하는 일: 행사장 약도를 그린다. 흰 배경에 길을 그리고, 길 위에 떨어져 놓인 네모 구역 A/B/C를
#   붐빔 단계의 색으로 칠하고, 찾은 사람은 점으로 찍는다.
# 해당 기능: 핵심 기능 1 (사람 세기)의 결과를 보여 주는 부분. 관리자 화면 요소 1번 (도면 위 구역별 붐빔 색).
# 해당 기준: C-16, C-31, C-43
# 근거: D-13 (관리자 화면에 넣을 것), D-35 (사람은 점), D-42 (구역은 네모), D-45 (사람 수로 단계를 나눔),
#   D-46 (약도의 모양: 구역은 떨어져 있고 길로 이어짐), D-12 (녹화 영상임을 표시), 01의 11번 (관측 불가 표시)
# 출처: AI 코딩 도구로 작성
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crowdflow.venue import Venue

MARGIN_TOP, MARGIN, BELOW = 44, 20, 44
ROAD, ROAD_EDGE = (214, 222, 235), (90, 96, 108)


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:  # 그 글꼴이 없는 컴퓨터
        return ImageFont.load_default()


def draw_venue(venue: Venue, points: dict[str, np.ndarray], status: dict[str, dict], view: dict, caption: str = "") -> Image.Image:
    """약도 그림을 돌려준다.

    points: 구역 → 그 구역 안 사람들의 약도 좌표 (N×2)
    status: 구역 → {"count", "count_avg", "level_index", "level_name", "density"(없으면 None)}
    view: settings.yaml의 plan_view
    """
    unit = view["pixels_per_unit"]
    W = int(venue.plan_size[0] * unit + 2 * MARGIN)
    H = int(venue.plan_size[1] * unit + MARGIN_TOP + BELOW)
    px = lambda u, v: (MARGIN + float(u) * unit, MARGIN_TOP + float(v) * unit)
    f_title, f_zone, f_text, f_small = (_font(view["font_bold"], 16), _font(view["font_bold"], 18),
                                        _font(view["font"], 14), _font(view["font"], 12))
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    dr.text((MARGIN, 8), f"{venue.name} 약도", font=f_title, fill="black")
    dr.text((MARGIN, 27), "시연용 설정 · 축척 없음", font=f_small, fill=(90, 90, 90))
    if caption:
        dr.text((W - MARGIN, 12), caption, font=f_text, fill=(60, 60, 60), anchor="ra")

    for u0, v0, w, h in venue.roads:  # 길: 테두리를 먼저 그리고 속을 채워, 길끼리 만나는 곳에 선이 남지 않게 한다
        dr.rectangle((*px(u0, v0), *px(u0 + w, v0 + h)), outline=ROAD_EDGE, width=3)
    for u0, v0, w, h in venue.roads:
        x0, y0 = px(u0, v0)
        x1, y1 = px(u0 + w, v0 + h)
        dr.rectangle((x0 + 3, y0 + 3, x1 - 3, y1 - 3), fill=ROAD)

    road_left = min((r[0] for r in venue.roads if r[3] > r[2]), default=0.0)  # 세로 길의 왼쪽 끝: 글은 그 왼쪽에 쓴다
    r = max(3, unit // 9)
    for k, z in venue.zones.items():
        s = status[k]
        u0, v0, w, h = z.plan_rect
        dr.rectangle((*px(u0, v0), *px(u0 + w, v0 + h)), fill=tuple(view["level_colors"][s["level_index"]]), outline="black", width=2)
        for u, v in np.asarray(points.get(k, np.empty((0, 2)))).reshape(-1, 2):  # 사람 점
            x, y = px(u, v)
            dr.ellipse((x - r, y - r, x + r, y + r), fill=(215, 40, 40))
        x, y = px(road_left - 0.3, v0 + h / 2)
        dr.text((x, y - 34), z.name, font=f_zone, fill="black", anchor="rm")
        dr.text((x, y - 10), z.street, font=f_text, fill=(70, 70, 70), anchor="rm")
        dr.text((x, y + 12), f"지금 {s['count']}명 · 평균 {s['count_avg']:.1f}명", font=f_text, fill="black", anchor="rm")
        dr.text((x, y + 34), s["level_name"], font=f_zone, fill="black", anchor="rm")
        if s.get("density") is not None:
            dr.text((x, y + 56), f"1㎡당 {s['density']:.2f}명 (넓이는 어림)", font=f_small, fill=(70, 70, 70), anchor="rm")

    y = H - BELOW + 14
    dr.text((MARGIN, y), "점: 구역 안에서 찾은 사람 · 구역 밖의 길은 관측 불가", font=f_small, fill=(90, 90, 90), anchor="lm")
    dr.text((MARGIN, y + 16), "녹화 영상으로 재현 (AI Hub 지능형 관제 서비스 CCTV 영상 데이터)", font=f_small, fill=(90, 90, 90), anchor="lm")
    return im
