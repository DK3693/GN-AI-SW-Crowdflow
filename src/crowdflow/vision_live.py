# 이 파일이 하는 일: 녹화된 CCTV 영상을 실제 시간대로 틀면서 그 순간의 장면에서 GPU로 사람을 찾아(YOLO) 구역별로 세고,
#   상자를 그린 영상(MJPEG)과 구역별 사람 수·약도 위의 사람 점(JSON)을 작은 HTTP 서버로 내준다.
#   detect_cache.py는 미리 분석해 저장해 두지만, 이 일꾼은 시연하는 그 자리에서 분석한다. 세는 방법과 설정은 같다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-23, C-30, C-37
# 근거: D-26 (미리 학습된 모델로 추론만), D-44 (주 자료), D-45 (영상의 차례, 5초 평균), D-12 (녹화 영상으로 재현),
#   D-55 (CCTV 영상과 검출 상자를 실시간으로 보여 줌), E-36 (모델), E-37 (발 위치로 구역 판정), E-45 (추론 기본값),
#   E-47 (세는 바닥, 고른 영상), E-57 (이 일꾼을 돌려 본 결과)
#   (사용자 지시 2026-10-04: CCTV 영상과 YOLO 결과를 실시간으로, GPU를 최대한 활용)
# 출처: AI 코딩 도구로 작성
#
# 실행: PYTHONPATH=src uv run --locked python -m crowdflow.vision_live --port 8502 --device 1
#   GET  /health            일꾼의 상태 (장치, 모델, 1초에 몇 번 분석하는지)
#   GET  /state             지금 시각, 구역별 사람 수·평균·약도 위의 점·영상 이름·프레임 번호
#   POST /clock             {"t": 12.3, "running": true} 로 영상 시계를 맞춘다. 시계의 주인은 흐름 엔진이다
#   GET  /stream/<구역>      상자를 그린 영상 (MJPEG). ?boxes=0 이면 상자 없는 영상
#   GET  /frame/<구역>.jpg   가장 최근 한 장. ?boxes=0 이면 상자 없는 사진
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np

from crowdflow.crowd_vision import load_yaml
from crowdflow.venue import Venue, load_venue

ROOT = Path(__file__).resolve().parents[2]

# ---- 이 파일에서 새로 생긴 설정 -------------------------------------------------------------------------
# 사람을 세는 값(모델, 입력 크기, 확신도, 몇 장마다 볼지, 평균 시간)은 모두 config/settings.yaml에서 읽는다.
# 아래는 화면과 운영에만 쓰는 값이다. AI 코딩 도구가 임시로 정했고, settings.yaml로 옮길 후보다.
DEFAULT_PORT = 8502            # 화면 서버(8501)의 옆 번호
DEFAULT_HOST = "127.0.0.1"     # 이 컴퓨터 안에서만 받는다. 다른 기기의 화면에서 영상을 보려면 --host 0.0.0.0
DEFAULT_DEVICE = 1             # GPU 번호 (nvidia-smi가 보여 주는 번호)
# 반정밀도(FP16)로 돌릴지. 직접 잰 값으로 골랐다 (2026-10-04, RTX 3090 한 장, 입력 크기 1280. E-57에 덧붙여 적을 값):
#   1초에 분석하는 횟수 — FP32: 구역 3개 6.9번, 4개 5.1번 (목표인 "구역 4개에서 8번 이상"에 못 미침) / FP16: 구역 3개 10.0번(모든 장면), 4개 9.2번
#   저장해 둔 분석(FP32)과 견준 구역 안 사람 수의 평균 차이 — FP32 0.02명, FP16 0.10명 (E05_026의 100장, 많아야 1명 차이).
#   3분 전체에서 FP16으로 붐빔 단계가 달라진 표본은 구역별로 0~0.5%였다.
DEFAULT_HALF = True
# FP16일 때 GPU 안에서 사진을 놓는 차례를 바꿔 더 빨리 돌린다 (Ultralytics의 channels_last). 구역 4개에서 1초에 8.2번 → 9.2번, 센 값의 차이는 그대로였다.
CHANNELS_LAST = True
DEFAULT_WIDTH = 960            # 내보내는 사진의 가로 픽셀. 세로는 16:9로 맞춘다
ASPECT = (16, 9)               # 내보내는 사진의 가로:세로
JPEG_QUALITY = 80              # 내보내는 사진의 JPEG 품질 (0~100)
IDLE_SECONDS = 1.0             # 시계가 멈춰 있을 때(시작 전, 영상이 끝난 뒤) 같은 장면을 다시 분석하는 간격(초)
POLL_SECONDS = 0.005           # 다음 장면이 될 때까지 시계를 다시 보는 간격(초)
HOLD_BACK_SECONDS = 1.0        # 도는 중에 시계가 이만큼까지 뒤로 간 것은 흔들림으로 보고 그 장면에 머문다(초)
RATE_WINDOW_SECONDS = 5.0      # "1초에 몇 번 분석하는지"를 지난 몇 초로 재는지
STREAM_WAIT_SECONDS = 1.0      # 영상을 보내는 쪽이 새 사진을 기다리다가 일꾼이 꺼졌는지 살펴보는 간격(초)
SOCKET_TIMEOUT_SECONDS = 15.0  # 보던 쪽이 이만큼 받지 못하면 연결을 끊는다(초)
ERROR_PAUSE_SECONDS = 1.0      # 분석하다 오류가 나면 이만큼 쉬고 다시 해 본다(초)
STALE_SECONDS = 10.0           # 이만큼 새 분석이 없으면 "멈춤"으로 보고 /health와 /state가 503으로 답한다(초)
MAX_BODY_BYTES = 4096          # POST로 받는 글의 최대 크기
BOUNDARY = b"crowdflowframe"   # MJPEG에서 사진과 사진 사이를 나누는 표시
# 그림의 색 [파랑, 초록, 빨강]과 선 굵기(픽셀). 밤 영상에서 잘 보이는 색으로 골랐다. 계산에는 쓰지 않는다.
FLOOR_COLOR, FLOOR_THICKNESS = (255, 200, 0), 2        # 세는 바닥의 테두리
COUNTED_COLOR, COUNTED_THICKNESS = (0, 255, 0), 2      # 센 사람 (발이 세는 바닥 안에 있음): 밝고 뚜렷하게
OUTSIDE_COLOR, OUTSIDE_THICKNESS = (150, 150, 150), 1  # 세지 않은 사람 (세는 바닥 밖): 흐리고 가늘게


# ---- 시각 → 영상과 프레임 (GPU도 영상 파일도 필요 없는 부분) ------------------------------------------------
def pick_frame(t: float, clip_seconds: float, clip_fps: Sequence[float], step: int) -> tuple[int, int]:
    """영상 시계의 시각 t(초) → (몇 번째 영상인지, 그 영상의 몇 번째 프레임인지).

    영상마다 clip_seconds초씩 쓰고 차례로 잇는다 (D-45). 프레임은 step장마다 한 장만 본다.
    그래서 detect_cache.py가 저장해 둔 것과 같은 프레임(0, 3, 6, ...)을 고르게 된다.
    맨 끝 시각에서는 마지막 영상의 마지막 프레임에 머문다.
    """
    k = min(max(int(t // clip_seconds), 0), len(clip_fps) - 1)
    local = max(t - k * clip_seconds, 0.0)  # 그 영상 안에서의 시각
    last = math.ceil(clip_seconds * clip_fps[k]) - 1  # 그 영상에서 쓰는 마지막 프레임 (60초 × 초당 30장이면 1799번)
    return k, (min(int(local * clip_fps[k]), last) // step) * step


class Clock:
    """영상 시계. 주인은 흐름 엔진이다. 엔진이 알려 준 기준 시각에서, 도는 동안에는 실제 시간만큼 흐른다."""

    def __init__(self, total: float, now: Callable[[], float] = time.monotonic):
        self.total = float(total)  # 영상 전체 길이(초). 시각은 0과 이 값 사이에 머문다
        self._now = now
        self._lock = threading.Lock()
        self._t, self._running, self._anchor = 0.0, False, now()

    def set(self, t: float, running: bool) -> None:
        """기준을 새로 잡는다: 지금이 영상의 t초이고, running이면 여기서부터 흐른다."""
        with self._lock:
            self._t, self._running, self._anchor = min(max(float(t), 0.0), self.total), bool(running), self._now()

    def read(self) -> tuple[float, bool]:
        """(지금이 영상의 몇 초인지, 도는 중인지). 멈춰 있으면 그 시각에 머물고, 영상이 끝나면 끝 시각에 머문다."""
        with self._lock:
            t = self._t + (self._now() - self._anchor) if self._running else self._t
            return min(max(t, 0.0), self.total), self._running


class WindowAverage:
    """지난 window초 동안의 사람 수 평균. 지난 값만 쓴다 (D-45, E-28).

    초당 10장으로 고르게 들어오면 timeline.py의 평균(지난 50장)과 같은 표본을 쓴다.
    """

    def __init__(self, window: float):
        self.window = float(window)
        self._samples: deque[tuple[float, int]] = deque()  # (영상 시각, 사람 수), 시각 순

    def add(self, t: float, count: int) -> float:
        s = self._samples
        while s and s[-1][0] >= t:  # 같은 시각이면 바꿔 넣는다. 시계가 뒤로 갔으면(처음부터 다시 등) 그 뒤의 표본은 버린다
            s.pop()
        s.append((t, count))
        # 창 밖으로 나간 옛 표본을 버린다. 딱 window초 전의 표본은 창 밖이다 (1e-9는 소수 계산의 끝자리 흔들림을 덮는 값)
        while len(s) > 1 and s[0][0] <= t - self.window + 1e-9:
            s.popleft()
        return sum(c for _, c in s) / len(s)

    def __len__(self) -> int:
        return len(self._samples)


# ---- 영상 읽기와 그림 ---------------------------------------------------------------------------------
class ClipReader:
    """영상 파일을 앞에서부터 차례로 읽는다. 보지 않는 장은 풀지 않고 넘기기만 한다(grab).

    건너뛰어 찾아가는 방법(seek)은 느리고 어긋날 수 있어서 쓰지 않는다 (detect_cache.py와 같음).
    다른 영상으로 바뀌거나 이미 지나간 장을 달라고 하면 파일을 처음부터 다시 연다.
    다만 가장 최근에 읽은 두 장은 들고 있어서, 다음 장면을 미리 읽어 둔 뒤에 지금 장면을 다시 달라고 해도 다시 열지 않는다.
    """

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._path: Path | None = None  # 지금 열어 둔 영상
        self._index = -1  # 그 영상에서 가장 최근에 읽은 장의 번호 (아직 없으면 -1)
        self._kept: list[tuple[Path, int, np.ndarray]] = []  # 가장 최근에 읽은 두 장 (영상, 번호, 사진), 새것이 뒤
        self._counts: dict[Path, int] = {}  # 영상마다 모두 몇 장인지 (파일에 적힌 값)

    @staticmethod
    def fps(path: Path) -> float:
        cap = cv2.VideoCapture(str(path))  # 한글 경로에서도 열린다 (cv2.imread와 다름)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        if not fps or not math.isfinite(fps):
            raise FileNotFoundError(f"영상 파일을 열 수 없음: {path}")
        return float(fps)

    def read(self, path: Path, index: int) -> tuple[np.ndarray | None, int]:
        """path 영상의 index번째 장과 그 번호. 영상이 그보다 짧으면 마지막 장에 머문다."""
        if self._counts.get(path):
            index = min(index, self._counts[path] - 1)
        for kept_path, kept_index, frame in self._kept:
            if kept_path == path and kept_index == index:
                return frame, index
        if path != self._path or index <= self._index:
            if self._cap is not None:
                self._cap.release()
            self._cap, self._path, self._index = cv2.VideoCapture(str(path)), path, -1
            self._counts[path] = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self._counts[path] > 0:
                index = min(index, self._counts[path] - 1)
        for _ in range(index - self._index - 1):
            if not self._cap.grab():
                break
        ok, frame = self._cap.read()
        if ok:
            self._index = index
            self._kept = [*self._kept[-1:], (path, index, frame)]
            return frame, index
        if self._kept and self._kept[-1][0] == path:  # 파일에 적힌 장수보다 일찍 끝남: 마지막으로 읽은 장에 머문다
            return self._kept[-1][2], self._kept[-1][1]
        self._path = None  # 파일을 열지 못했거나 첫 장도 읽지 못함. 다음에 다시 열어 본다
        return None, -1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap, self._path, self._index, self._kept = None, None, -1, []


def _jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("사진을 JPEG으로 바꾸지 못함")
    return buf.tobytes()


def render_pictures(frame: np.ndarray, floor_quad_px: np.ndarray, xyxy: np.ndarray, inside: np.ndarray,
                    size: tuple[int, int]) -> tuple[bytes, bytes]:
    """한 장면 → (상자 없는 JPEG, 상자를 그린 JPEG). 둘 다 size(가로, 세로)로 줄인다.

    그리는 것: 세는 바닥의 테두리, 센 사람의 상자(밝게), 세지 않은 사람의 상자(흐리게).
    글자는 넣지 않는다. "녹화 영상으로 재현" 같은 글은 화면 쪽에서 얹는다 (D-12).
    """
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    raw = _jpeg(small)
    scale = np.array([size[0] / frame.shape[1], size[1] / frame.shape[0]])
    quad = np.round(np.asarray(floor_quad_px, dtype=float) * scale).astype(np.int32)
    cv2.polylines(small, [quad], True, FLOOR_COLOR, FLOOR_THICKNESS, cv2.LINE_AA)
    boxes = np.round(np.asarray(xyxy, dtype=float).reshape(-1, 4) * np.tile(scale, 2)).astype(int)
    inside = np.asarray(inside, dtype=bool)
    for counted, colour, thickness in ((False, OUTSIDE_COLOR, OUTSIDE_THICKNESS), (True, COUNTED_COLOR, COUNTED_THICKNESS)):
        for x1, y1, x2, y2 in boxes[inside == counted].tolist():  # 세지 않은 사람을 먼저, 센 사람을 그 위에 그린다
            cv2.rectangle(small, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    return raw, _jpeg(small)


# ---- 사람 찾기 (GPU) ----------------------------------------------------------------------------------
class YoloDetector:
    """미리 학습된 YOLO 모델로 여러 장면에서 한꺼번에 사람을 찾는다 (D-26). 설정은 detect_cache.py와 같다."""

    def __init__(self, cfg: dict, half: bool):
        import torch  # 무거운 것은 여기서 부른다. GPU 번호를 먼저 정해야 하고(main), 시험도 빨리 돌게 하려는 것이다
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU를 쓸 수 없음 (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        weights = ROOT / cfg["model"]
        if not weights.exists():  # 없으면 Ultralytics가 내려받으려 한다. 여기서는 내려받지 않는다
            raise FileNotFoundError(f"모델 파일이 없음: {weights}")
        self._torch, self.cfg, self.half = torch, cfg, bool(half)
        self.gpu_name = torch.cuda.get_device_name(0)
        self.model = YOLO(str(weights))
        # Ultralytics 8.4.171에서는 half 대신 quantize=16이 반정밀도(FP16)다 (설치한 버전의 cfg/default.yaml).
        # FP32로 돌릴 때는 detect_cache.py와 똑같이 부른다 (저장해 둔 분석과 가장 가깝게).
        self._extra = {"quantize": 16, **({"channels_last": True} if CHANNELS_LAST else {})} if self.half else {}

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """장면들 → 장면마다 사람 상자(N×4, 원래 화면의 픽셀 x1 y1 x2 y2)."""
        c = self.cfg
        results = self.model.predict(frames, classes=[c["person_class_id"]], conf=c["conf"], iou=c["iou"], imgsz=c["imgsz"],
                                     verbose=False, device=0, **self._extra)
        return [r.boxes.xyxy.cpu().numpy().astype(np.float32) for r in results]

    def memory_mb(self) -> float:
        return round(self._torch.cuda.memory_reserved(0) / 2**20, 1)


# ---- 일꾼 --------------------------------------------------------------------------------------------
@dataclass
class ZoneView:
    """구역 한 곳의 가장 최근 분석 결과."""

    count: int  # 세는 바닥 안에서 찾은 사람 수 (그 장면 그대로)
    count_avg: float  # 지난 몇 초의 평균 (붐빔 단계는 이 값으로 정한다, D-45)
    points: list[list[float]]  # 센 사람들의 약도 좌표
    clip: str  # 영상 이름
    frame: int  # 그 영상의 몇 번째 프레임인지


class VisionWorker:
    """영상 시계를 따라가며 구역마다 한 장씩 읽고, 한꺼번에 사람을 찾고, 결과와 사진을 내놓는다.

    detect는 장면들을 받아 장면마다 사람 상자를 돌려주는 함수다 (실제로는 YoloDetector).
    모델을 올리는 동안에도 서버가 "준비 중"이라고 답할 수 있게, detect는 나중에 넣어도 된다.

    GPU가 사람을 찾는 동안 다른 일을 겹쳐서 한다 (1초에 분석하는 횟수를 늘리려고):
    방금 분석한 장면의 사진(줄이기·그리기·JPEG)을 만들고, 다음 장면을 미리 읽어 둔다.
    그래서 사람 수가 먼저 나오고 그 장면의 사진은 조금(천분의 몇 초) 뒤에 나온다.
    """

    def __init__(self, settings: dict, venue: Venue, detect: Callable[[list[np.ndarray]], list[np.ndarray]] | None = None,
                 width: int = DEFAULT_WIDTH, info: dict | None = None, reader: type = ClipReader,
                 now: Callable[[], float] = time.monotonic):
        self.venue, self.detect, self.info, self._now = venue, detect, dict(info or {}), now
        self.step = int(settings["detection"]["video_step"])
        self.size = (int(width), int(round(width * ASPECT[1] / ASPECT[0])))
        n_clips = min(len(z.cctv.clips) for z in venue.zones.values())  # 구역마다 영상 수가 다르면 적은 쪽에 맞춘다
        self.total = n_clips * venue.clip_seconds
        self.clock = Clock(self.total, now)
        self.fps = {k: [reader.fps(z.cctv.video(c)) for c in z.cctv.clips[:n_clips]] for k, z in venue.zones.items()}
        # 새 장면이 나오는 빠르기: 초당 30장을 3장마다 보면 1초에 10번. 이보다 자주 분석하지 않는다
        self.tick_rate = max(f for v in self.fps.values() for f in v) / self.step
        self._readers = {k: reader() for k in venue.zones}
        self._averages = {k: WindowAverage(settings["crowd_levels"]["smoothing_seconds"]) for k in venue.zones}
        # 영상 읽기와 사진 만들기를 맡는 일손들. 구역마다 읽기 하나와 사진 하나가 동시에 돌 수 있게 구역 수의 두 배를 둔다
        self._pool = ThreadPoolExecutor(max_workers=2 * len(venue.zones), thread_name_prefix="vision-io")
        self._ahead: list[Future] = []  # 다음 장면을 미리 읽고 있는 일들
        self._cond = threading.Condition()  # 아래 값들을 바꾸거나 읽을 때, 새 사진을 기다릴 때 쓴다
        self._views: dict[str, ZoneView] = {}
        self._pictures: dict[str, tuple[int, bytes, bytes]] = {}  # 구역 → (몇 번째 분석의 사진인지, 상자 없는 JPEG, 상자를 그린 JPEG)
        self._seq = 0  # 분석을 한 번 끝낼 때마다 1씩 오른다
        self._stamps: deque[float] = deque()  # 분석을 끝낸 때들 (빠르기를 잴 때 씀)
        self._sample_t: float | None = None  # 가장 최근에 분석한 장면의 영상 시각
        self._tick, self._analysed_at = -1, 0.0
        self.stage_ms: dict[str, float] = {}  # 가장 최근 분석에서 단계마다 든 시간(천분의 1초). 느려졌을 때 어디서 느린지 보려는 것
        self.error: str | None = None  # 가장 최근 분석이 실패했으면 그 까닭
        self.render_error: str | None = None  # 가장 최근 사진 만들기가 실패했으면 그 까닭
        self.closed = False

    # ---- 분석 ----
    def poll(self) -> bool:
        """새 장면을 볼 때가 됐으면 한 번 분석하고 True를 돌려준다."""
        t, running = self.clock.read()
        if running and self._sample_t is not None and 0 < self._sample_t - t <= HOLD_BACK_SECONDS:
            t = self._sample_t  # 엔진이 시계를 다시 맞추다 아주 조금 뒤로 간 것은 따라가지 않는다 (영상을 처음부터 다시 읽지 않으려고)
        if int(t * self.tick_rate) == self._tick and self._now() - self._analysed_at < IDLE_SECONDS:
            return False
        self.analyse(t, ahead=running)
        return True

    def analyse(self, t: float, ahead: bool = False) -> None:
        """영상 시각 t의 장면을 구역마다 한 장씩 읽어 사람을 찾고, 결과를 새로 내놓는다. 사진은 뒤에서 만들어 곧 내놓는다.

        ahead가 True이면(시계가 도는 중) 바로 다음 장면을 미리 읽기 시작한다.
        """
        keys = list(self.venue.zones)
        picks = [pick_frame(t, self.venue.clip_seconds, self.fps[k], self.step) for k in keys]
        started = time.perf_counter()
        wait_futures(self._ahead)  # 미리 읽던 일이 끝난 뒤에 읽는다 (한 영상을 두 곳에서 동시에 읽지 않게)
        frames = list(self._pool.map(self._read, keys, picks))  # 구역마다 한 장씩 동시에. 미리 읽어 둔 장이면 바로 나온다
        if ahead:  # GPU가 일하는 동안 다음 장면(시계가 다음 칸으로 넘어가는 때의 장면)을 읽어 둔다
            t_next = min((int(t * self.tick_rate) + 1) / self.tick_rate + 1e-6, self.total)
            self._ahead = [self._pool.submit(self._read, k, pick_frame(t_next, self.venue.clip_seconds, self.fps[k], self.step))
                           for k in keys]
        read_done = time.perf_counter()
        boxes = self.detect([f for f, _ in frames])  # 모든 구역의 장면을 한 묶음으로 GPU에 넣는다
        detect_done = time.perf_counter()
        views, jobs = {}, []
        for k, (clip_i, _), (frame, shown), xyxy in zip(keys, picks, frames, boxes):
            zone = self.venue.zones[k]
            unit, inside = zone.cctv.locate(xyxy)  # 발이 세는 바닥 안에 있는 사람만 센다 (E-37)
            count = int(inside.sum())
            views[k] = ZoneView(count, self._averages[k].add(t, count), np.round(zone.to_plan(unit[inside]), 3).tolist(),
                                zone.cctv.clips[clip_i], int(shown))
            jobs.append((k, frame, zone.cctv.floor_quad_px, xyxy, inside))
        done = self._now()
        with self._cond:
            self._views = views
            self._seq = seq = self._seq + 1
            self._sample_t, self._tick, self._analysed_at = t, int(t * self.tick_rate), done
            self._stamps.append(done)
            while self._stamps[0] < done - RATE_WINDOW_SECONDS:
                self._stamps.popleft()
            self.stage_ms = {**self.stage_ms, "read": round((read_done - started) * 1000, 1),
                             "detect": round((detect_done - read_done) * 1000, 1)}
        for job in jobs:  # 사진(줄이기·그리기·JPEG)은 구역마다 따로, 다음 분석과 겹쳐서 만든다
            self._pool.submit(self._render, seq, *job)

    def _read(self, key: str, pick: tuple[int, int]) -> tuple[np.ndarray, int]:
        cctv = self.venue.zones[key].cctv
        frame, shown = self._readers[key].read(cctv.video(cctv.clips[pick[0]]), pick[1])
        if frame is None:
            raise RuntimeError(f"{key}구역의 영상을 읽지 못함: {cctv.clips[pick[0]]}")
        return frame, shown

    def _render(self, seq: int, key: str, frame: np.ndarray, floor_quad_px: np.ndarray, xyxy: np.ndarray, inside: np.ndarray) -> None:
        """seq번째 분석의 사진을 만들어 내놓는다. 보고 있는 쪽(MJPEG)에 새 사진이 나왔다고 알린다."""
        started = time.perf_counter()
        try:
            raw, boxed = render_pictures(frame, floor_quad_px, xyxy, inside, self.size)
        except Exception as e:  # 사진 한 장이 안 만들어져도 일꾼은 계속 돈다. 까닭은 /health에 보인다
            self.render_error = f"{type(e).__name__}: {e}"[:300]
            return
        with self._cond:
            if seq > self._pictures.get(key, (0, b"", b""))[0]:
                self._pictures[key] = (seq, raw, boxed)
            self.stage_ms = {**self.stage_ms, "render": round((time.perf_counter() - started) * 1000, 1)}
            self.render_error = None
            self._cond.notify_all()

    def run(self) -> None:
        """멈추라고 할 때까지 돈다. 한 번의 분석이 실패해도 멈추지 않고, 까닭을 적어 두고 다시 해 본다."""
        while not self.closed:
            try:
                busy = self.poll()
            except Exception as e:  # GPU 오류, 영상 읽기 오류 등
                if self.closed:
                    break
                message = f"{type(e).__name__}: {e}"[:300]
                if message != self.error:
                    print(f"vision_live: analysis failed: {message}", file=sys.stderr, flush=True)
                self.error = message
                time.sleep(ERROR_PAUSE_SECONDS)
                continue
            if busy:
                self.error = None
            else:
                time.sleep(POLL_SECONDS)

    def close(self) -> None:
        self.closed = True
        with self._cond:
            self._cond.notify_all()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- 서버가 읽는 것 ----
    @property
    def ready(self) -> bool:
        """첫 분석이 끝나고 모든 구역의 사진이 나왔는지."""
        return self._seq > 0 and len(self._pictures) == len(self.venue.zones)

    def wait_pictures(self, timeout: float) -> bool:
        """가장 최근 분석의 사진이 모든 구역에서 나올 때까지 기다린다."""
        with self._cond:
            return self._cond.wait_for(
                lambda: len(self._pictures) == len(self.venue.zones) and all(p[0] == self._seq for p in self._pictures.values()), timeout)

    def rate(self) -> float:
        """지난 몇 초 동안 1초에 몇 번 분석했는지. 한 번에 모든 구역을 한 장씩 분석한다."""
        now = self._now()
        with self._cond:
            n = sum(1 for s in self._stamps if now - s <= RATE_WINDOW_SECONDS)
        return round(n / RATE_WINDOW_SECONDS, 2)

    def problem(self) -> str | None:
        """지금 내는 값을 믿을 수 없으면 그 까닭, 괜찮으면 None. 까닭이 있으면 /health와 /state가 503으로 답한다.

        그러면 받는 쪽(흐름 엔진)이 옛 값을 새 값으로 잘못 알지 않고, 미리 분석해 둔 결과로 돌아갈 수 있다.
        """
        if not self.ready:
            return "준비 중 (모델을 올리는 중)"
        if self.error or self.render_error:
            return self.error or self.render_error
        if self._now() - self._analysed_at > STALE_SECONDS:
            return f"분석이 {STALE_SECONDS:g}초 넘게 멈춰 있음"
        return None

    def health(self) -> dict:
        memory = getattr(self.detect, "memory_mb", None)
        problem = self.problem()
        return {"ok": problem is None, **self.info, "rate": self.rate(), "zones": list(self.venue.zones),
                "iterations": self._seq, "total": self.total, "width": self.size[0], "pid": os.getpid(),
                "gpu_memory_mb": memory() if memory else None, "stage_ms": self.stage_ms, "error": problem}

    def state(self) -> dict:
        t, running = self.clock.read()
        with self._cond:
            views, sample_t, analysed_at = self._views, self._sample_t, self._analysed_at
        return {"t": round(t, 3), "running": running, "rate": self.rate(),
                "zones": {k: {"count": v.count, "count_avg": v.count_avg, "points": v.points, "clip": v.clip, "frame": v.frame}
                          for k, v in views.items()},
                # 아래는 덧붙인 것: 사람 수가 영상의 몇 초 장면에서 나왔는지, 그 분석이 몇 초 전에 끝났는지, 영상 전체 길이
                "sample_t": None if sample_t is None else round(sample_t, 3), "age": round(self._now() - analysed_at, 3),
                "total": self.total}

    def picture(self, key: str, boxes: bool = True) -> bytes | None:
        """그 구역의 가장 최근 사진. boxes가 False이면 상자 없는 사진."""
        picture = self._pictures.get(key)
        return None if picture is None else picture[2 if boxes else 1]

    def wait_picture(self, key: str, boxes: bool, after: int, timeout: float) -> tuple[int, bytes | None]:
        """after번째 분석보다 새로운 사진이 나올 때까지 기다린다. 나오면 (번호, 사진), 시간 안에 안 나오면 (after, None)."""
        with self._cond:
            self._cond.wait_for(lambda: self.closed or self._pictures.get(key, (0,))[0] > after, timeout)
            picture = self._pictures.get(key)
            if self.closed or picture is None or picture[0] <= after:
                return after, None
            return picture[0], picture[2 if boxes else 1]


# ---- HTTP 서버 (Python에 들어 있는 것만 씀) -------------------------------------------------------------
class _Server(ThreadingHTTPServer):
    daemon_threads = True  # 영상을 보내던 연결이 남아 있어도 일꾼을 끌 수 있게
    request_queue_size = 64  # 화면이 영상 여러 개를 한꺼번에 열어도 받는다
    # Windows에서는 이 값을 켜면 이미 쓰는 포트에 또 붙을 수 있어서, 예전 일꾼이 남아 있어도 모르고 지나간다. 그래서 끈다.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], OSError):  # 보던 쪽이 연결을 끊은 것. 조용히 넘어간다
            return
        super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    worker: VisionWorker  # make_server가 넣는다
    server_version = "CrowdFlowVision"
    timeout = SOCKET_TIMEOUT_SECONDS

    def log_message(self, format, *args):  # 요청마다 한 줄씩 찍지 않는다
        pass

    def _json(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")  # 화면(다른 포트)의 스크립트도 읽을 수 있게
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _zone(self, key: str) -> bool:
        """구역 이름이 맞고 첫 분석이 끝났는지. 아니면 까닭을 답하고 False."""
        if key not in self.worker.venue.zones:
            self._json({"ok": False, "error": f"없는 구역: {key}", "zones": list(self.worker.venue.zones)}, 404)
        elif not self.worker.ready:
            self._json({"ok": False, "error": "준비 중 (모델을 올리는 중)"}, 503)
        else:
            return True
        return False

    def do_GET(self):
        url = urlsplit(self.path)
        path, boxes = unquote(url.path), parse_qs(url.query).get("boxes", ["1"])[0] != "0"
        worker = self.worker
        try:
            if path == "/health":
                health = worker.health()
                self._json(health, 200 if health["ok"] else 503)
            elif path == "/state":
                problem = worker.problem()
                self._json(worker.state() if problem is None else {"ok": False, "error": problem}, 200 if problem is None else 503)
            elif path.startswith("/stream/"):
                if self._zone(path[len("/stream/"):]):
                    self._stream(path[len("/stream/"):], boxes)
            elif path.startswith("/frame/") and path.endswith(".jpg"):
                if self._zone(path[len("/frame/"):-len(".jpg")]):
                    self._picture(worker.picture(path[len("/frame/"):-len(".jpg")], boxes))
            else:
                self._json({"ok": False, "error": f"없는 주소: {path}"}, 404)
        except OSError:  # 보던 쪽이 연결을 끊음 (브라우저 탭을 닫음 등)
            self.close_connection = True

    def _picture(self, jpeg: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def _stream(self, key: str, boxes: bool) -> None:
        """분석이 한 번 끝날 때마다 사진 한 장을 보낸다 (MJPEG). 보던 쪽이 끊으면 OSError가 나고 do_GET이 조용히 끝낸다."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(b"--" + BOUNDARY + b"\r\n")
        seq = -1
        while not self.worker.closed:
            seq, jpeg = self.worker.wait_picture(key, boxes, seq, STREAM_WAIT_SECONDS)
            if jpeg is None:
                continue
            # 사진 바로 뒤에 다음 나눔 표시까지 보낸다. 그래야 브라우저가 다음 사진을 기다리지 않고 바로 그린다
            self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg
                             + b"\r\n--" + BOUNDARY + b"\r\n")

    def do_POST(self):
        try:
            length = self.headers.get("Content-Length") or ""
            length = int(length) if length.isdigit() else 0
            raw = self.rfile.read(length) if 0 < length <= MAX_BODY_BYTES else b""  # 답하기 전에 받은 글을 먼저 다 읽는다
            if urlsplit(self.path).path != "/clock":
                return self._json({"ok": False, "error": f"없는 주소: {self.path}"}, 404)
            try:
                body = json.loads(raw)
                t, running = body["t"], body["running"]
                if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not isinstance(running, bool):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                return self._json({"ok": False, "error": '{"t": 숫자, "running": true 또는 false} 꼴로 보내야 함'}, 400)
            self.worker.clock.set(t, running)
            self._json({"ok": True})
        except OSError:
            self.close_connection = True

    def do_OPTIONS(self):  # 브라우저의 스크립트가 POST를 보내기 전에 먼저 묻는 것에 답한다
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()


def make_server(worker: VisionWorker, host: str, port: int) -> ThreadingHTTPServer:
    return _Server((host, port), type("Handler", (_Handler,), {"worker": worker}))


def main() -> None:
    p = argparse.ArgumentParser(description="CCTV 영상을 실시간으로 분석해 내주는 일꾼")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="GPU 번호 (nvidia-smi의 번호)")
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=DEFAULT_HALF, help="반정밀도(FP16)로 돌린다")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="내보내는 사진의 가로 픽셀 (세로는 16:9)")
    args = p.parse_args()
    if args.width < 16:
        p.error("--width는 16 이상이어야 함")

    # 고른 GPU 한 장만 보이게 한다. torch가 GPU를 처음 건드리기 전에 정해야 해서 맨 앞에 둔다.
    # 번호는 nvidia-smi가 보여 주는 번호와 같게 한다. 다른 GPU(AI 모델이 올라가 있음)는 건드리지 않는다.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # Ultralytics가 아무것도 내려받지 않고, 쓰는 기록(익명 통계)도 밖으로 보내지 않게 한다 (E-06).
    os.environ.setdefault("YOLO_OFFLINE", "true")

    settings = load_yaml(ROOT / "config" / "settings.yaml")
    venue = load_venue(ROOT / settings["venue"])
    info = {"device": f"cuda:{args.device}", "model": settings["detection"]["model"], "half": bool(args.half)}
    worker = VisionWorker(settings, venue, width=args.width, info=info)
    try:
        server = make_server(worker, args.host, args.port)  # 모델을 올리기 전에 포트부터 잡는다 (이미 쓰이고 있으면 바로 알 수 있게)
    except OSError as e:
        sys.exit(f"vision_live: cannot listen on {args.host}:{args.port} ({e})")
    threading.Thread(target=server.serve_forever, daemon=True, name="vision-http").start()
    print(f"vision_live: http://{args.host}:{args.port} zones {','.join(venue.zones)} total {worker.total:g}s, loading model ...", flush=True)
    try:
        worker.detect = YoloDetector(settings["detection"], half=args.half)
        worker.info["gpu"] = worker.detect.gpu_name
        worker.poll()  # 첫 장면을 분석해 둔다. 여기서 모델이 GPU에 올라간다 (실패하면 켜지 않고 끝낸다)
        worker.wait_pictures(SOCKET_TIMEOUT_SECONDS)
        print(f"vision_live: ready on {info['device']} ({worker.detect.gpu_name}), half={info['half']}, "
              f"imgsz {settings['detection']['imgsz']}", flush=True)
        worker.run()
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
