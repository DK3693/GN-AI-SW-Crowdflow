# 이 파일이 하는 일: 저장해 둔 사람 상자(detect_cache.py의 결과)를 읽어, 구역마다 영상을 차례로 이은 시간표를 만들고,
#   시각마다 약도 위의 사람 점, 구역별 사람 수, 붐빔 단계를 낸다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-02, C-23, C-24, C-30, C-37
# 근거: D-35, D-37·D-45 (붐빔 단계), D-44 (주 자료), D-45 (영상의 차례, 평균 시간), E-27, E-28, E-47
# 출처: AI 코딩 도구로 작성
#
# 실행: PYTHONPATH=src uv run python -m crowdflow.timeline   (결과: outputs/timeline/zone_status.csv)
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crowdflow.crowd_vision import load_yaml, moving_average
from crowdflow.levels import level_for
from crowdflow.venue import Venue, Zone, load_venue

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ZoneTrack:
    """구역 한 곳의 시간표: 표본(시각)마다 그 구역의 세는 바닥 안에 있는 사람들의 약도 좌표."""

    zone: Zone
    sample_rate: float  # 1초에 표본이 몇 개인지
    clip: np.ndarray  # (S,) 표본마다 몇 번째 영상인지
    frame: np.ndarray  # (S,) 표본마다 그 영상의 몇 번째 프레임인지
    offsets: np.ndarray  # (S+1,) 표본 i의 점은 points[offsets[i]:offsets[i+1]]
    points: np.ndarray  # (M, 2) 약도 좌표

    def at(self, i: int) -> np.ndarray:
        return self.points[self.offsets[i]:self.offsets[i + 1]]

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)


def load_track(zone: Zone, clip_seconds: float) -> ZoneTrack:
    clip_idx, frame_idx, sizes, pts = [], [], [], []
    rate = None
    for k, clip in enumerate(zone.cctv.clips):
        d = np.load(zone.cctv.detections(clip))
        # 초당 표본 수. 영상은 초당 30장 또는 29.97장이라(E-47) 3장마다 보면 10 또는 9.99가 된다.
        # 0.1%의 차이라 둘 다 10으로 본다.
        rate = float(round(float(d["fps"]) / int(d["step"])))
        seen = d["frames_seen"][: int(round(clip_seconds * rate))]  # 영상마다 같은 수의 표본만 쓴다
        order = np.argsort(d["frame"], kind="stable")
        frames, xyxy = d["frame"][order], d["xyxy"][order]
        starts, ends = np.searchsorted(frames, seen), np.searchsorted(frames, seen, side="right")
        for f, a, b in zip(seen, starts, ends):
            unit, inside = zone.cctv.locate(xyxy[a:b])
            p = zone.to_plan(unit[inside])
            clip_idx.append(k)
            frame_idx.append(int(f))
            sizes.append(len(p))
            pts.append(p)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return ZoneTrack(zone, rate, np.array(clip_idx), np.array(frame_idx), offsets, np.vstack(pts) if pts else np.empty((0, 2)))


@dataclass
class Timeline:
    venue: Venue
    tracks: dict[str, ZoneTrack]
    sample_rate: float
    n_samples: int  # 모든 구역에 공통으로 있는 표본 수
    counts: dict[str, np.ndarray]  # 구역 → 표본별 사람 수 (그 순간에 찾은 그대로)
    averaged: dict[str, np.ndarray]  # 구역 → 지난 몇 초의 평균 사람 수 (흔들림을 줄인 값)

    def seconds(self, i: int) -> float:
        return i / self.sample_rate

    def status(self, i: int, levels_cfg: dict) -> dict[str, dict]:
        """표본 i에서의 구역별 상태. 단계는 평균 사람 수로 정한다 (D-45)."""
        out = {}
        for k, z in self.venue.zones.items():
            avg = float(self.averaged[k][i])
            lv = level_for(avg, levels_cfg)
            out[k] = {
                "count": int(self.counts[k][i]), "count_avg": avg, "level_index": lv.index, "level_name": lv.name,
                "area_m2": z.area_m2, "density": (avg / z.area_m2) if z.area_m2 else None,
            }
        return out


def build_timeline(venue: Venue, smoothing_seconds: float) -> Timeline:
    tracks = {k: load_track(z, venue.clip_seconds) for k, z in venue.zones.items()}
    rates = {t.sample_rate for t in tracks.values()}
    assert len(rates) == 1, f"구역마다 표본 간격이 다름: {rates}"
    rate = rates.pop()
    n = min(len(t.counts) for t in tracks.values())
    counts = {k: t.counts[:n] for k, t in tracks.items()}
    window = max(1, int(round(smoothing_seconds * rate)))
    return Timeline(venue, tracks, rate, n, counts, {k: moving_average(c, window) for k, c in counts.items()})


def main():
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    cfg = settings["crowd_levels"]
    venue = load_venue(ROOT / settings["venue"])
    tl = build_timeline(venue, cfg["smoothing_seconds"])
    out = ROOT / "outputs" / "timeline"
    out.mkdir(parents=True, exist_ok=True)
    step = int(round(tl.sample_rate))  # 1초마다 한 줄
    with open(out / "zone_status.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["second", "zone", "clip", "count", "count_avg", "level"])
        for i in range(0, tl.n_samples, step):
            for k, s in tl.status(i, cfg).items():
                t = tl.tracks[k]
                w.writerow([round(tl.seconds(i)), k, venue.zones[k].cctv.clips[t.clip[i]], s["count"], round(s["count_avg"], 1), s["level_name"]])
    print(f"{tl.n_samples} samples at {tl.sample_rate:g}/s = {tl.n_samples / tl.sample_rate:.0f} s, average over {cfg['smoothing_seconds']} s")
    for k in venue.zones:
        avg = tl.averaged[k]
        idx = np.array([level_for(float(v), cfg).index for v in avg])
        names = [cfg["below_name"]] + [lv["name"] for lv in cfg["levels"]]
        share = " ".join(f"{nm} {np.mean(idx == j) * 100:.0f}%" for j, nm in enumerate(names))
        on = idx >= 1
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        spans = [f"{a / tl.sample_rate:.0f}~{(b - 1) / tl.sample_rate:.0f}초" for a, b in zip(edges[::2], edges[1::2])]
        print(f"zone {k} ({venue.zones[k].street}): average {avg.mean():.1f}, max {avg.max():.1f} | {share} | {names[1]} 이상인 구간: {spans or '없음'}")
    print("->", out / "zone_status.csv")


if __name__ == "__main__":
    main()
