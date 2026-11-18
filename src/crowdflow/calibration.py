# 이 파일이 하는 일: 카메라 보정 자료(Tsai 방식, ETISEO XML)를 읽어 화면의 점을 땅 위의 점(미터)으로
#   바꾸고, 화면에 그린 구역의 실제 넓이(㎡)를 계산한다.
# 해당 기능: 핵심 기능 1 (사람 세기 — 1㎡당 사람 수를 내는 데 필요)
# 해당 기준: C-02, C-24
# 근거: E-30 (QUT 보정 자료로 넓이 계산), E-42 (자료에 든 calibration.xml), D-30 (직접 재지 않음), D-34
# 출처: AI 코딩 도구로 작성. 식은 Tsai 카메라 모델(자료에 든 Ryan 2012 논문이 쓰는 방식)을 따름.
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TsaiCamera:
    dpx: float  # 픽셀 하나의 가로 크기 (센서 단위)
    dpy: float
    focal: float
    kappa1: float  # 렌즈 휨
    cx: float
    cy: float
    sx: float
    R: np.ndarray  # 3×3, 세계 좌표 → 카메라 좌표
    T: np.ndarray  # (3,)

    @classmethod
    def from_etiseo_xml(cls, path: str | Path) -> "TsaiCamera":
        root = ET.parse(path).getroot().find("ETISEOCameraModel")
        g, i, e = (root.find(k).attrib for k in ("geometry", "intrinsic", "extrinsic"))
        rx, ry, rz = (float(e[k]) for k in ("rx", "ry", "rz"))
        ca, sa, cb, sb, cg, sg = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
        R = np.array(
            [
                [cb * cg, cg * sa * sb - ca * sg, sa * sg + ca * cg * sb],
                [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - cg * sa],
                [-sb, cb * sa, ca * cb],
            ]
        )
        return cls(
            dpx=float(g["dpx"]), dpy=float(g["dpy"]), focal=float(i["focal"]), kappa1=float(i["kappa1"]),
            cx=float(i["cx"]), cy=float(i["cy"]), sx=float(i["sx"]),
            R=R, T=np.array([float(e[k]) for k in ("tx", "ty", "tz")]),
        )

    @property
    def position(self) -> np.ndarray:
        """카메라의 세계 좌표 (x, y, 높이)."""
        return -self.R.T @ self.T

    def image_to_ground(self, u, v, z: float = 0.0) -> np.ndarray:
        """화면의 점(u, v) → 높이 z인 평면 위의 점 (x, y). 여러 점을 한 번에 넣을 수 있다."""
        u, v = np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float))
        xd = self.dpx * (u - self.cx) / self.sx
        yd = self.dpy * (v - self.cy)
        k = 1.0 + self.kappa1 * (xd * xd + yd * yd)
        rays = np.stack([xd * k, yd * k, np.full_like(xd, self.focal)], axis=1) @ self.R  # 각 행 = R^T · d
        origin = self.R.T @ self.T
        t = (z + origin[2]) / rays[:, 2]
        return (rays * t[:, None] - origin)[:, :2]

    def ground_to_image(self, x, y, z: float = 0.0) -> np.ndarray:
        """세계의 점 (x, y, z) → 화면의 점 (u, v)."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        pc = np.stack([x, y, np.full_like(x, z)], axis=1) @ self.R.T + self.T
        xu, yu = self.focal * pc[:, 0] / pc[:, 2], self.focal * pc[:, 1] / pc[:, 2]
        xd, yd = xu.copy(), yu.copy()
        for _ in range(20):  # 렌즈 휨을 되돌린다
            k = 1.0 + self.kappa1 * (xd * xd + yd * yd)
            xd, yd = xu / k, yu / k
        return np.stack([self.sx * xd / self.dpx + self.cx, yd / self.dpy + self.cy], axis=1)


def polygon_area(points: np.ndarray) -> float:
    """다각형의 넓이 (신발끈 공식)."""
    p = np.asarray(points, float)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def image_polygon_ground_area(cam: TsaiCamera, polygon_px, samples_per_edge: int = 50) -> float:
    """화면에 그린 다각형이 땅 위에서 차지하는 넓이(㎡).

    화면의 직선이 땅 위에서는 조금 휘므로(렌즈 휨), 변을 잘게 나눠서 옮긴다.
    """
    poly = np.asarray(polygon_px, float)
    pts = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        s = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)[:, None]
        pts.append(a + s * (b - a))
    pts = np.vstack(pts)
    return polygon_area(cam.image_to_ground(pts[:, 0], pts[:, 1]))
