# 이 파일이 하는 일: 실시간 영상 분석 일꾼(vision_live.py)이 맞게 도는지 확인한다.
#   앞쪽 시험은 GPU도 영상 자료도 없이 돈다 (시각 → 프레임, 지난 몇 초의 평균, 영상 시계, 그림, 시험 안에서 만든 작은 영상 읽기,
#   가짜 영상과 가짜 사람 찾기로 돌린 일꾼과 서버). 맨 끝의 시험 하나만 실제 일꾼을 켜 본다. GPU나 영상 파일이 없는 컴퓨터에서는 건너뛴다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-48 (안정성·시험)
# 근거: D-44, D-45, E-37, E-47 (사용자 지시 2026-10-04: CCTV 영상과 YOLO 결과를 실시간으로)
# 출처: AI 코딩 도구로 작성
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import wait as wait_futures
from pathlib import Path

import cv2
import numpy as np
import pytest

from crowdflow import vision_live as vl
from crowdflow.crowd_vision import load_yaml, moving_average
from crowdflow.venue import Cctv, Venue, Zone, load_venue

ROOT = Path(__file__).parent.parent
SETTINGS = {"detection": {"video_step": 3}, "crowd_levels": {"smoothing_seconds": 5}}  # 일꾼이 읽는 두 값 (settings.yaml과 같은 값)
FRAME = np.full((1080, 1920, 3), 40, dtype=np.uint8)  # 가짜 영상의 한 장 (어두운 회색)


# ---- 시각 → 프레임, 평균, 시계 ---------------------------------------------------------------------------
def test_time_picks_the_frames_the_cache_uses():
    fps = [30.0, 30.0, 30000 / 1001]  # 셋째 영상은 초당 29.97장 (E-47)
    assert vl.pick_frame(0.0, 60, fps, 3) == (0, 0)
    assert vl.pick_frame(12.34, 60, fps, 3) == (0, 369)  # 12.34초 × 30장 = 370번째 → 3장마다 보므로 369
    assert vl.pick_frame(60.05, 60, fps, 3) == (1, 0)  # 60초가 지나면 둘째 영상의 처음
    assert vl.pick_frame(119.99, 60, fps, 3) == (1, 1797)
    assert vl.pick_frame(179.95, 60, fps, 3) == (2, 1794)  # 59.95초 × 29.97장 = 1796번째 → 1794
    assert vl.pick_frame(180.0, 60, fps, 3) == (2, 1797)  # 끝 시각: 마지막 영상의 마지막 장에 머문다
    assert vl.pick_frame(999.0, 60, fps, 3) == (2, 1797) and vl.pick_frame(-1.0, 60, fps, 3) == (0, 0)
    for t in np.arange(0, 180, 0.037):  # 언제나 3의 배수이고, 영상마다 60초 안의 장이다 (저장해 둔 분석과 같은 장들)
        k, f = vl.pick_frame(float(t), 60, fps, 3)
        assert k == int(t // 60) and f % 3 == 0 and 0 <= f <= 1797


def test_average_uses_only_the_last_seconds():
    counts = [(i * 7) % 23 for i in range(200)]
    avg = vl.WindowAverage(5)
    got = [avg.add(i / 10, c) for i, c in enumerate(counts)]  # 초당 10장으로 고르게 들어올 때
    assert got == pytest.approx(moving_average(np.array(counts), 50))  # 저장해 둔 분석의 평균(지난 50장)과 같다
    assert len(avg) == 50


def test_average_forgets_what_comes_after_when_the_clock_goes_back():
    avg = vl.WindowAverage(5)
    for i in range(30):
        avg.add(100 + i / 10, 30)
    assert avg.add(0.0, 7) == 7 and len(avg) == 1  # 처음으로 되돌리면 지난 기록을 버린다
    assert avg.add(0.0, 9) == 9 and len(avg) == 1  # 같은 시각의 장면을 다시 분석하면 바꿔 넣는다
    avg.add(1.0, 10)
    avg.add(1.1, 20)
    assert avg.add(1.05, 30) == pytest.approx((9 + 10 + 30) / 3)  # 조금 뒤로 가면 그 뒤의 것(1.1초)만 버린다


def test_clock_holds_runs_and_stops_at_the_ends():
    now = [100.0]
    clock = vl.Clock(180.0, now=lambda: now[0])
    assert clock.read() == (0.0, False)
    now[0] += 7
    assert clock.read() == (0.0, False)  # 시작 전에는 흐르지 않는다
    clock.set(12.3, True)
    now[0] += 2.5
    assert clock.read() == (pytest.approx(14.8), True)  # 도는 동안에는 실제 시간만큼 흐른다
    clock.set(12.3, False)
    now[0] += 2.5
    assert clock.read() == (12.3, False)  # 멈추면 그 시각에 머문다
    clock.set(170, True)
    now[0] += 60
    assert clock.read() == (180.0, True)  # 영상이 끝나면 끝 시각에 머문다
    clock.set(-5, False)
    assert clock.read() == (0.0, False)
    clock.set(999, True)
    assert clock.read() == (180.0, True)
    clock.set(0, True)  # 처음부터 다시
    now[0] += 1
    assert clock.read() == (pytest.approx(1.0), True)


# ---- 그림 ---------------------------------------------------------------------------------------------
def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_pictures_have_boxes_only_in_the_annotated_version():
    quad = np.array([[744, 204], [1212, 204], [1548, 1056], [396, 1056]], dtype=float)
    boxes = np.array([[900, 500, 1000, 800], [100, 200, 200, 500]], dtype=np.float32)  # 한 명은 세는 바닥 안, 한 명은 밖
    raw, boxed = vl.render_pictures(FRAME, quad, boxes, np.array([True, False]), (960, 540))
    a, b = decode(raw).astype(int), decode(boxed).astype(int)
    assert a.shape == b.shape == (540, 960, 3)
    assert abs(a - 40).max() <= 3  # 상자 없는 사진에는 아무것도 그리지 않는다
    blue, green, red = b[325, 450]  # 센 사람의 상자(왼쪽 변): 밝은 초록
    assert green > 180 and green > blue + 80 and green > red + 80
    assert (b[175, 50] > 90).all() and b[175, 50].max() < 200  # 세지 않은 사람의 상자: 흐린 회색
    assert b[102, 500, 0] > 180 and b[102, 500, 2] < 90  # 세는 바닥의 테두리(먼 쪽 선)
    assert abs(b[20, 20] - 40).max() <= 3 and abs(b[325, 475] - 40).max() <= 3  # 그 밖은 그대로. 글자도 넣지 않는다


# ---- 영상 읽기 (시험 안에서 만든 작은 영상으로) -------------------------------------------------------------
def make_video(path: Path, frames: int) -> bool:
    """장마다 밝기가 다른 작은 영상을 만든다 (밝기로 몇 번째 장인지 알 수 있게). 만들지 못하면 False."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    if not writer.isOpened():
        return False
    for i in range(frames):
        writer.write(np.full((48, 64, 3), 20 + 5 * i, dtype=np.uint8))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def frame_number(frame: np.ndarray) -> int:
    return round((float(frame.mean()) - 20) / 5)


def test_reader_reads_in_order_and_keeps_the_last_two(tmp_path):
    a, b = tmp_path / "a.avi", tmp_path / "b.avi"
    if not (make_video(a, 40) and make_video(b, 10)):
        pytest.skip("이 컴퓨터의 OpenCV로는 시험용 영상을 만들 수 없음")
    assert vl.ClipReader.fps(a) == pytest.approx(30.0)
    with pytest.raises(FileNotFoundError):
        vl.ClipReader.fps(tmp_path / "no_such.avi")
    reader = vl.ClipReader()
    try:
        for index in [0, 3, 6, 15]:  # 앞으로 갈 때는 사이의 장을 넘기면서 차례로 읽는다
            frame, shown = reader.read(a, index)
            assert (shown, frame_number(frame)) == (index, index)
        opened = reader._cap
        for index in [6, 15, 18, 15]:  # 가장 최근의 두 장은 들고 있어서, 다시 달라고 해도 파일을 다시 열지 않는다
            frame, shown = reader.read(a, index)
            assert (shown, frame_number(frame)) == (index, index) and reader._cap is opened
        frame, shown = reader.read(a, 3)  # 이미 지나간 장: 처음부터 다시 열어서 읽는다
        assert (shown, frame_number(frame)) == (3, 3) and reader._cap is not opened
        frame, shown = reader.read(a, 500)  # 영상이 그보다 짧으면 마지막 장(39번)에 머문다
        assert (shown, frame_number(frame)) == (39, 39)
        frame, shown = reader.read(b, 0)  # 다음 영상으로 넘어가도
        assert (shown, frame_number(frame)) == (0, 0)
        opened = reader._cap
        frame, shown = reader.read(a, 500)  # 앞 영상의 마지막 장면은 아직 들고 있다 (다음 장면을 미리 읽어 둔 뒤에 멈춘 경우)
        assert (shown, frame_number(frame)) == (39, 39) and reader._cap is opened
        assert reader.read(tmp_path / "no_such.avi", 0) == (None, -1)  # 없는 영상
        frame, shown = reader.read(b, 4)
        assert (shown, frame_number(frame)) == (4, 4)
    finally:
        reader.close()


# ---- 가짜 영상과 가짜 사람 찾기로 돌리는 일꾼 ----------------------------------------------------------------
class FakeReader:
    """영상 파일 대신 쓴다: 빈 사진을 주고, 무엇을 달라고 했는지와 파일을 몇 번 새로 열었는지 적어 둔다."""

    opens = 0
    asked: list[tuple[str, int]] = []  # (영상 이름, 프레임 번호)

    def __init__(self):
        self.path, self.index = None, -1

    @staticmethod
    def fps(path):
        return 30.0

    def read(self, path, index):
        if path != self.path or index < self.index:
            FakeReader.opens += 1
        FakeReader.asked.append((path.stem, index))
        self.path, self.index = path, index
        return FRAME, index


class FakeDetect:
    """사람 찾기 대신 쓴다: 구역마다 세는 바닥 안에 people명, 바닥 밖(먼 쪽 선보다 위)에 1명을 돌려준다."""

    def __init__(self, venue: Venue, people: int = 3):
        self.venue, self.people, self.calls = venue, people, 0

    def __call__(self, frames):
        self.calls += 1
        out = []
        for zone in list(self.venue.zones.values())[: len(frames)]:
            quad = zone.cctv.floor_quad_px
            cx, cy = quad.mean(axis=0)
            feet = [(cx + 12 * i, cy) for i in range(self.people)] + [(cx, quad[0][1] - 40)]
            out.append(np.array([[x - 30, y - 200, x + 30, y] for x, y in feet], dtype=np.float32))
        return out


def make_venue() -> Venue:
    """구역 두 곳. 이름이 A, B, C가 아니어도 되고, 구역마다 영상 수가 달라도 된다 (적은 쪽에 맞춘다)."""
    def zone(key, quad, rect, clips):
        cctv = Cctv(id=f"CCTV-{key}", image_size=(1920, 1080), floor_quad_px=np.array(quad, dtype=float), clips=clips,
                    video_dir=Path("no_videos"), detections_dir=Path("no_cache"))
        return Zone(key=key, name=f"{key}구역", street="시험 골목", plan_rect=rect, area_m2=None, cctv=cctv)

    return Venue(name="시험 행사장", plan_size=(11.5, 18.8), roads=[], clip_seconds=60.0, zones={
        "A": zone("A", [[744, 204], [1212, 204], [1548, 1056], [396, 1056]], (6.5, 1.0, 3.6, 5.5), ["a1", "a2"]),
        "D": zone("D", [[800, 312], [1188, 312], [1128, 1056], [20, 1056]], (0.5, 16.0, 3.6, 2.5), ["d1", "d2", "d3"]),
    })


def make_worker(now=time.monotonic, people: int = 3) -> vl.VisionWorker:
    venue = make_venue()
    info = {"device": "가짜", "model": "가짜", "half": False}
    return vl.VisionWorker(SETTINGS, venue, FakeDetect(venue, people), info=info, reader=FakeReader, now=now)


def test_worker_counts_people_on_the_floor_and_follows_the_clock():
    now = [0.0]
    worker = make_worker(now=lambda: now[0])
    try:
        assert worker.total == 120 and not worker.ready and worker.tick_rate == 10  # 영상 두 개 × 60초, 1초에 10장면
        assert worker.poll() is True  # 시작 전에도 0초의 장면을 분석해 둔다
        s = worker.state()
        assert (s["t"], s["running"], s["sample_t"], list(s["zones"])) == (0.0, False, 0.0, ["A", "D"])
        for key, z in s["zones"].items():
            assert (z["count"], z["count_avg"], z["frame"], z["clip"]) == (3, 3, 0, key.lower() + "1")  # 바닥 밖의 1명은 세지 않는다
            u0, v0, w, h = worker.venue.zones[key].plan_rect  # 사람 점은 자기 구역 네모 안에 찍힌다
            assert len(z["points"]) == 3 and all(u0 <= u <= u0 + w and v0 <= v <= v0 + h for u, v in z["points"])
        assert worker.poll() is False  # 같은 장면이면 다시 분석하지 않는다
        now[0] += vl.IDLE_SECONDS
        assert worker.poll() is True  # 멈춰 있어도 가끔 다시 분석한다
        assert worker.state()["zones"]["A"]["count_avg"] == 3  # 같은 장면을 여러 번 봐도 평균에는 한 번만 들어간다

        worker.detect.people = 5
        worker.clock.set(12.34, True)  # 엔진이 시작을 알림
        assert worker.poll() is True
        s = worker.state()
        assert (s["running"], s["zones"]["D"]["frame"], s["zones"]["D"]["count"], s["zones"]["D"]["count_avg"]) == (True, 369, 5, 5)
        wait_futures(worker._ahead)  # 도는 동안에는 바로 다음 장면(12.4초의 372번)을 미리 읽어 둔다
        assert set(FakeReader.asked[-4:-2]) == {("a1", 369), ("d1", 369)} and set(FakeReader.asked[-2:]) == {("a1", 372), ("d1", 372)}
        now[0] += 0.05
        assert worker.poll() is False  # 아직 같은 장면 (12.39초)
        now[0] += 0.05
        assert worker.poll() is True and worker.state()["zones"]["A"]["frame"] == 372  # 다음 장면 (12.44초)
        now[0] += 60
        assert worker.poll() is True
        s = worker.state()
        assert (s["zones"]["A"]["clip"], s["zones"]["D"]["clip"], s["zones"]["A"]["frame"]) == ("a2", "d2", 372)  # 둘째 영상으로
        now[0] += 1000  # 영상이 끝난 뒤: 마지막 장면에 머문다
        assert worker.poll() is True and worker.poll() is False
        s = worker.state()
        assert (s["t"], s["running"], s["zones"]["D"]["clip"], s["zones"]["D"]["frame"]) == (120.0, True, "d2", 1797)
        now[0] += vl.IDLE_SECONDS
        assert worker.poll() is True and worker.state()["zones"]["A"]["frame"] == 1797
        assert worker.wait_pictures(5) and worker.ready  # 사진은 뒤에서 만들어져 곧 나온다
        assert decode(worker.picture("A")).shape == decode(worker.picture("D", boxes=False)).shape == (540, 960, 3)
        assert worker.health()["ok"] is True and worker.health()["iterations"] == worker.detect.calls == 7
    finally:
        worker.close()


def test_small_step_back_is_held_and_a_reset_clears_the_average():
    now = [0.0]
    worker = make_worker(now=lambda: now[0], people=10)
    try:
        worker.clock.set(30.0, True)
        for _ in range(16):  # 2초 동안 10명 (0.125초씩: 소수 계산이 딱 떨어지는 간격)
            assert worker.poll() is True
            now[0] += 0.125
        assert worker.state()["zones"]["A"]["count_avg"] == 10
        opens, calls, sample_t = FakeReader.opens, worker.detect.calls, worker.state()["sample_t"]
        worker.clock.set(sample_t - 0.2, True)  # 엔진이 시계를 다시 맞추다 아주 조금 뒤로 감
        assert worker.poll() is False and worker.detect.calls == calls  # 그 장면에 머문다
        assert FakeReader.opens == opens  # 영상을 처음부터 다시 읽지 않는다
        now[0] += 0.35
        assert worker.poll() is True and worker.state()["sample_t"] > sample_t  # 시계가 따라잡으면 다시 나아간다

        worker.detect.people = 2
        worker.clock.set(0.0, False)  # 처음부터 다시
        assert worker.poll() is True
        s = worker.state()
        assert (s["t"], s["zones"]["A"]["frame"], s["zones"]["A"]["count"], s["zones"]["A"]["count_avg"]) == (0.0, 0, 2, 2)  # 10명의 기록이 섞이지 않는다
        assert FakeReader.opens == opens + 2  # 두 구역의 영상을 처음부터 다시 연다
    finally:
        worker.close()


# ---- 서버 (가짜 영상, 가짜 사람 찾기) ---------------------------------------------------------------------
def fetch(url: str, body: dict | bytes | None = None, timeout: float = 5.0):
    """(상태 번호, 머리말, 내용). 오류 답(404, 503 등)도 예외 없이 그대로 돌려준다."""
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


@pytest.fixture
def served():
    """가짜 일꾼과 서버를 켠다. 돌려주는 것: (일꾼, 주소, 포트)."""
    worker = make_worker()
    server = vl.make_server(worker, "127.0.0.1", 0)  # 0: 비어 있는 포트를 알아서 고른다
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield worker, f"http://127.0.0.1:{port}", port
    finally:
        worker.close()
        server.shutdown()
        server.server_close()


def run_in_background(worker: vl.VisionWorker) -> None:
    threading.Thread(target=worker.run, daemon=True).start()
    deadline = time.time() + 5
    while not worker.ready and time.time() < deadline:
        time.sleep(0.01)
    assert worker.ready


def test_server_says_loading_until_the_first_analysis(served):
    worker, base, _ = served
    for path in ["/health", "/state", "/frame/A.jpg", "/stream/A"]:
        status, _, body = fetch(base + path)
        assert status == 503 and json.loads(body)["ok"] is False
    status, _, body = fetch(base + "/clock", {"t": 3, "running": False})  # 시계는 준비 전에도 맞출 수 있다
    assert status == 200 and json.loads(body) == {"ok": True} and worker.clock.read() == (3.0, False)


def test_server_answers_state_pictures_and_clock(served):
    worker, base, _ = served
    run_in_background(worker)
    status, headers, body = fetch(base + "/health")
    health = json.loads(body)
    assert status == 200 and headers["Access-Control-Allow-Origin"] == "*"
    assert health["ok"] is True and health["zones"] == ["A", "D"] and {"device", "model", "half", "rate"} <= set(health)

    status, headers, body = fetch(base + "/state")
    state = json.loads(body)
    assert status == 200 and headers["Access-Control-Allow-Origin"] == "*"
    assert state["t"] == 0 and state["running"] is False and set(state["zones"]) == {"A", "D"}
    assert set(state["zones"]["A"]) == {"count", "count_avg", "points", "clip", "frame"} and state["zones"]["A"]["count"] == 3

    status, headers, boxed = fetch(base + "/frame/D.jpg")
    assert status == 200 and headers["Content-Type"] == "image/jpeg" and decode(boxed).shape == (540, 960, 3)
    status, _, raw = fetch(base + "/frame/D.jpg?boxes=0")
    assert status == 200 and abs(decode(raw).astype(int) - 40).max() <= 3 and decode(boxed).max() > 150  # 상자는 한쪽에만 있다

    status, _, body = fetch(base + "/clock", {"t": 20.0, "running": True})
    assert status == 200 and json.loads(body) == {"ok": True}
    time.sleep(0.5)
    state = json.loads(fetch(base + "/state")[2])
    assert state["running"] is True and 20.3 < state["t"] < 22 and state["sample_t"] > 20 and state["zones"]["A"]["frame"] >= 600
    assert state["rate"] > 0 and state["age"] < 1

    for bad in [{"t": "열두 시", "running": True}, {"t": 1}, {"running": True}, {"t": True, "running": True}, b"{not json", b""]:
        status, _, body = fetch(base + "/clock", bad)
        assert status == 400 and json.loads(body)["ok"] is False
    assert fetch(base + "/frame/Z.jpg")[0] == 404 and fetch(base + "/stream/Z")[0] == 404 and fetch(base + "/nothing")[0] == 404
    assert fetch(base + "/state", {"t": 1, "running": True})[0] == 404  # POST는 /clock만 받는다
    assert json.loads(fetch(base + "/state")[2])["running"] is True  # 잘못된 요청이 시계를 바꾸지 않았다


def wait_for_status(url: str, status: int, seconds: float = 5.0) -> bytes:
    deadline = time.time() + seconds
    while True:
        got, _, body = fetch(url)
        if got == status:
            return body
        assert time.time() < deadline, f"{url}: {status}을(를) 기다렸지만 {got}"
        time.sleep(0.05)


def test_failed_analysis_is_reported_and_old_numbers_are_not_served_as_new(served):
    worker, base, _ = served
    run_in_background(worker)
    good = worker.detect

    def broken(frames):
        raise RuntimeError("GPU가 답하지 않음")

    worker.detect = broken
    fetch(base + "/clock", {"t": 0, "running": True})
    assert "GPU가 답하지 않음" in json.loads(wait_for_status(base + "/state", 503))["error"]  # 옛 값을 새 값처럼 내주지 않는다
    health = json.loads(wait_for_status(base + "/health", 503))
    assert health["ok"] is False and "GPU가 답하지 않음" in health["error"]
    assert fetch(base + "/frame/A.jpg")[0] == 200  # 마지막 사진은 계속 볼 수 있다
    worker.detect = good  # 다시 되면 스스로 돌아온다
    assert set(json.loads(wait_for_status(base + "/state", 200))["zones"]) == {"A", "D"}
    assert json.loads(fetch(base + "/health")[2])["error"] is None


def test_stalled_analysis_is_reported():
    now = [0.0]
    worker = make_worker(now=lambda: now[0])
    try:
        assert worker.problem() == "준비 중 (모델을 올리는 중)"
        assert worker.poll() is True and worker.wait_pictures(5) and worker.problem() is None
        now[0] += vl.STALE_SECONDS + 1  # 분석이 오래 멈춰 있으면 (GPU가 멈춘 경우 등)
        assert "멈춰" in worker.problem() and worker.health()["ok"] is False
        assert worker.poll() is True and worker.problem() is None  # 다시 분석하면 풀린다
    finally:
        worker.close()


def read_stream(port: int, path: str, parts: int) -> tuple[socket.socket, bytes]:
    """영상 주소에 붙어서 사진 parts장이 올 때까지 읽는다. (열려 있는 연결, 받은 내용)."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: test\r\n\r\n".encode())
    data = b""
    while data.count(b"--" + vl.BOUNDARY) < parts + 1:  # 맨 앞에 하나, 사진마다 뒤에 하나
        chunk = sock.recv(1 << 16)
        assert chunk, "서버가 영상을 보내다 연결을 닫음"
        data += chunk
    return sock, data


def test_stream_sends_pictures_and_survives_a_closed_tab(served, capfd):
    worker, base, port = served
    run_in_background(worker)
    fetch(base + "/clock", {"t": 0, "running": True})
    sock, data = read_stream(port, "/stream/A", parts=3)
    head, _, rest = data.partition(b"\r\n\r\n")
    assert b"200" in head.split(b"\r\n")[0] and b"multipart/x-mixed-replace; boundary=" + vl.BOUNDARY in head
    part_head, _, part = rest.partition(b"\r\n\r\n")
    assert part_head.startswith(b"--" + vl.BOUNDARY) and b"Content-Type: image/jpeg" in part_head
    length = int(part_head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
    assert decode(part[:length]).shape == (540, 960, 3) and part[length:].startswith(b"\r\n--" + vl.BOUNDARY)
    # 브라우저 탭을 닫은 것처럼 갑자기 끊는다
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("hh", 1, 0))
    sock.close()
    time.sleep(0.5)
    assert json.loads(fetch(base + "/health")[2])["ok"] is True  # 일꾼은 계속 돈다
    sock, data = read_stream(port, "/stream/D?boxes=0", parts=2)  # 다시 붙으면 다시 보낸다
    sock.close()
    worker.clock.set(0.0, False)  # 시계가 멈추고 뒤로 가도 영상은 끊기지 않는다 (같은 장면을 가끔 다시 보낸다)
    sock, data = read_stream(port, "/stream/A", parts=2)
    sock.close()
    time.sleep(0.3)
    assert "Traceback" not in capfd.readouterr().err  # 끊긴 연결 때문에 오류를 찍지 않는다


# ---- 실제 일꾼 (GPU와 영상 파일이 있어야 함) ---------------------------------------------------------------
def test_real_worker_starts_analyses_and_serves(tmp_path):
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    venue = load_venue(ROOT / settings["venue"])
    clips = [z.cctv.video(c) for z in venue.zones.values() for c in z.cctv.clips]
    if not all(p.exists() for p in clips) or not (ROOT / settings["detection"]["model"]).exists():
        pytest.skip("영상 파일이나 모델 파일이 이 컴퓨터에 없음 (저장소에 올리지 않음)")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU(CUDA)가 없음")
    device = min(vl.DEFAULT_DEVICE, torch.cuda.device_count() - 1)
    with socket.socket() as s:  # 비어 있는 포트를 하나 고른다
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    log = open(tmp_path / "worker.log", "wb")
    proc = subprocess.Popen([sys.executable, "-m", "crowdflow.vision_live", "--port", str(port), "--device", str(device)],
                            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    pid = None
    try:
        deadline = time.time() + 180  # 모델을 GPU에 올리는 데 시간이 걸린다
        while True:
            assert proc.poll() is None and time.time() < deadline, (tmp_path / "worker.log").read_text(encoding="utf-8", errors="replace")
            try:
                status, _, body = fetch(base + "/health", timeout=2)
            except OSError:  # 아직 포트를 열지 않음
                status = None
            if status == 200:
                break
            time.sleep(0.5)
        health = json.loads(body)
        pid = health["pid"]
        assert health["ok"] is True and health["zones"] == list(venue.zones) and health["device"] == f"cuda:{device}"
        assert health["model"] == settings["detection"]["model"] and isinstance(health["half"], bool)

        state = json.loads(fetch(base + "/state")[2])
        assert state["t"] == 0 and state["running"] is False and list(state["zones"]) == list(venue.zones)
        for key, z in state["zones"].items():
            assert z["clip"] == venue.zones[key].cctv.clips[0] and z["frame"] == 0
            assert z["count"] == len(z["points"]) > 0 and z["count_avg"] == z["count"]  # 고른 영상에는 처음부터 사람이 있다 (E-47)
        key = next(iter(venue.zones))
        status, headers, jpeg = fetch(base + f"/frame/{key}.jpg")
        assert status == 200 and headers["Content-Type"] == "image/jpeg"
        assert decode(jpeg).shape == (round(vl.DEFAULT_WIDTH * 9 / 16), vl.DEFAULT_WIDTH, 3)

        assert json.loads(fetch(base + "/clock", {"t": 5.0, "running": True})[2]) == {"ok": True}
        time.sleep(2.0)
        state = json.loads(fetch(base + "/state")[2])
        assert state["running"] is True and state["t"] > 6.5 and state["rate"] > 0
        assert all(z["frame"] > 150 for z in state["zones"].values())  # 5초부터 흘러서 150번째 장을 지났다
    finally:
        proc.terminate()
        proc.wait(timeout=20)
        if pid is not None:  # 가상 환경의 python.exe가 진짜 파이썬을 따로 띄운 경우에 대비해, 그것도 끈다
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        log.close()
    deadline = time.time() + 10  # 끈 뒤에는 포트에 아무도 남아 있지 않아야 한다
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        except OSError:
            break
        time.sleep(0.2)
    else:
        pytest.fail("일꾼을 껐는데도 포트가 열려 있음")
