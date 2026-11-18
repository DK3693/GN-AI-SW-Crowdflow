# 이 파일이 하는 일: 행사장 약도, 세는 바닥 판정, 시간표, 약도 그리기가 맞는지 확인한다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-48 (안정성·시험)
# 근거: D-35, D-42, D-44, D-45, E-37, E-47
# 출처: AI 코딩 도구로 작성
from pathlib import Path

import numpy as np
import pytest

from crowdflow.crowd_vision import load_yaml
from crowdflow.plan_view import draw_venue
from crowdflow.timeline import build_timeline
from crowdflow.venue import load_venue

ROOT = Path(__file__).parent.parent
SETTINGS = load_yaml(ROOT / "config" / "settings.yaml")
# 기준값은 여기서 고정한다. 시연용 기준(config/settings.yaml)이 바뀌어도 이 시험은 그대로다
SETTINGS["crowd_levels"]["levels"] = [{"name": "주의", "min": 25}, {"name": "경계", "min": 30}, {"name": "심각", "min": 35}]
SETTINGS["venue"] = "tests/data/venue_three_zones.yaml"  # 구역 셋짜리 행사장으로 고정
VENUE = ROOT / SETTINGS["venue"]
HAS_CACHE = (ROOT / "outputs" / "detections" / "aihub" / "E05_026.npz").exists()
needs_cache = pytest.mark.skipif(not HAS_CACHE, reason="영상 분석 결과가 이 컴퓨터에 없음 (저장소에 올리지 않음)")


def box_with_feet_at(x, y, w=60, h=200):
    return [x - w / 2, y - h, x + w / 2, y]


def test_zones_are_separate_and_joined_by_the_road():
    # D-46: 구역은 세로 길을 따라 위에서부터 A, B, C로 떨어져 놓이고, 길로 이어진다
    venue = load_venue(VENUE)
    assert list(venue.zones) == ["A", "B", "C"]  # D-11: 구역 3개
    rects = [venue.zones[k].plan_rect for k in "ABC"]
    for (_, v0, _, h), (_, v1, _, _) in zip(rects, rects[1:]):
        assert v0 + h < v1  # 위 구역의 아래 끝과 다음 구역의 위 끝 사이가 떨어져 있다
    main_road = max(venue.roads, key=lambda r: r[3])  # 세로 길
    for u0, v0, w, h in rects:  # 모든 구역이 세로 길 위에 있다
        assert main_road[0] <= u0 and u0 + w <= main_road[0] + main_road[2]
        assert main_road[1] <= v0 and v0 + h <= main_road[1] + main_road[3]
    assert all(r[0] >= 0 and r[1] >= 0 and r[0] + r[2] <= venue.plan_size[0] and r[1] + r[3] <= venue.plan_size[1] for r in venue.roads)
    assert all(len(z.cctv.clips) == 3 for z in venue.zones.values())


def test_foot_position_decides_inside_and_plan_position():
    venue = load_venue(VENUE)
    zone = venue.zones["A"]
    quad = zone.cctv.floor_quad_px  # 먼 왼쪽, 먼 오른쪽, 가까운 오른쪽, 가까운 왼쪽
    centre = quad.mean(axis=0)
    boxes = np.array([
        box_with_feet_at(*centre),  # 발이 바닥 한가운데 → 안
        box_with_feet_at(quad[3][0] - 80, quad[3][1] - 10),  # 발이 가까운 왼쪽 귀퉁이의 바깥 → 밖
        box_with_feet_at(centre[0], quad[0][1] - 40),  # 발이 먼 쪽 선보다 위 → 밖 (몸은 바닥에 걸쳐 있어도)
        box_with_feet_at(centre[0], 1200),  # 상자가 화면 아래에서 잘림 → 화면 끝(1079)을 발로 봄 → 세는 바닥(1056까지)의 밖
        box_with_feet_at(centre[0], quad[2][1] - 5),  # 발이 가까운 쪽 선 바로 안 → 안
    ])
    unit, inside = zone.cctv.locate(boxes)
    assert inside.tolist() == [True, False, False, False, True]
    assert 0.2 < unit[0, 0] < 0.8 and 0.2 < unit[0, 1] < 0.9
    # 화면의 네 귀퉁이는 구역 네모의 네 귀퉁이로 간다: 먼 쪽이 위, 가까운 쪽이 아래
    corners_unit, _ = zone.cctv.locate(np.array([box_with_feet_at(x, y) for x, y in quad]))
    assert corners_unit == pytest.approx(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), abs=1e-3)
    u0, v0, w, h = zone.plan_rect
    assert zone.to_plan(corners_unit) == pytest.approx(np.array([[u0, v0], [u0 + w, v0], [u0 + w, v0 + h], [u0, v0 + h]]), abs=1e-2)


def test_no_people():
    venue = load_venue(VENUE)
    unit, inside = venue.zones["B"].cctv.locate(np.empty((0, 4)))
    assert unit.shape == (0, 2) and inside.shape == (0,)


@needs_cache
def test_timeline_joins_three_clips_and_levels_follow_settings():
    venue = load_venue(VENUE)
    cfg = SETTINGS["crowd_levels"]
    tl = build_timeline(venue, cfg["smoothing_seconds"])
    assert tl.sample_rate == pytest.approx(10.0)  # 초당 30장을 3장마다
    assert tl.n_samples == 3 * 60 * 10  # 영상 세 개 × 60초
    for k, t in tl.tracks.items():
        assert t.clip[0] == 0 and t.clip[tl.n_samples - 1] == 2
        assert (t.counts[: tl.n_samples] == tl.counts[k]).all()
        # 점은 모두 자기 구역 네모 안에 찍힌다
        u0, v0, w, h = venue.zones[k].plan_rect
        p = t.points
        assert (p[:, 0] >= u0).all() and (p[:, 0] <= u0 + w).all() and (p[:, 1] >= v0).all() and (p[:, 1] <= v0 + h).all()
    # E-47에서 잰 값: B구역은 둘째 영상(60~120초)에서 붐비고, 그 앞뒤에서는 한산하다
    b = tl.averaged["B"]
    assert b[300:580].max() < 25 and b[800:1150].min() >= 25 and b[1600:1790].max() < 25
    status = tl.status(900, cfg)  # 90초
    assert status["B"]["level_name"] in ("주의", "경계") and status["A"]["level_name"] == "평상" and status["C"]["level_name"] == "평상"
    assert status["B"]["density"] is None  # 넓이를 모르므로 1㎡당 사람 수는 내지 않는다


@needs_cache
def test_drawing():
    venue = load_venue(VENUE)
    cfg = SETTINGS["crowd_levels"]
    tl = build_timeline(venue, cfg["smoothing_seconds"])
    img = draw_venue(venue, {k: t.at(900) for k, t in tl.tracks.items()}, tl.status(900, cfg), SETTINGS["plan_view"], caption="1분 30초")
    assert img.size[1] > img.size[0] > 300  # 세로로 긴 약도 (D-46)
    assert img.getpixel((5, 5)) == (255, 255, 255)  # 흰 배경
