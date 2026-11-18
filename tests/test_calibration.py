# 이 파일이 하는 일: 화면 ↔ 땅 좌표 바꾸기와 넓이 계산이 맞는지 확인한다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-48 (안정성·시험)
# 근거: E-30
# 출처: AI 코딩 도구로 작성
from pathlib import Path

import numpy as np
import pytest

from crowdflow.calibration import TsaiCamera, image_polygon_ground_area, polygon_area

QUT_B = Path(__file__).parent.parent / "data/qut/SAIVT-QUTCrowdCountingDatabase/Datasets/QUT/B"


def make_camera() -> TsaiCamera:
    """시험용으로 만든 카메라: 원점 위 5 높이에서 똑바로 아래를 본다. 실제 값이 아니다."""
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return TsaiCamera(dpx=0.01, dpy=0.01, focal=5.0, kappa1=0.0, cx=320.0, cy=240.0, sx=1.0,
                      R=R, T=np.array([0.0, 0.0, 5.0]))


def test_polygon_area_unit_square():
    assert polygon_area(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) == pytest.approx(1.0)


def test_straight_down_camera_scale():
    cam = make_camera()
    assert cam.position == pytest.approx([0.0, 0.0, 5.0])
    # 높이 5, 초점 5, 픽셀 0.01 → 땅에서 한 픽셀이 0.01
    assert cam.image_to_ground(320, 240)[0] == pytest.approx([0.0, 0.0])
    assert cam.image_to_ground(420, 240)[0] == pytest.approx([1.0, 0.0])
    # 100×100 픽셀 네모 = 땅에서 1×1
    square = [[320, 240], [420, 240], [420, 340], [320, 340]]
    assert image_polygon_ground_area(cam, square) == pytest.approx(1.0)


def test_round_trip():
    cam = make_camera()
    cam.kappa1 = 0.002
    g = cam.image_to_ground([100, 500], [50, 400])
    back = cam.ground_to_image(g[:, 0], g[:, 1])
    assert back == pytest.approx(np.array([[100, 50], [500, 400]]), abs=1e-6)


@pytest.mark.skipif(not (QUT_B / "calibration.xml").exists(), reason="QUT 자료가 이 컴퓨터에 없음 (저장소에 올리지 않음)")
def test_qut_camera_b_roi_area_matches_e30():
    cam = TsaiCamera.from_etiseo_xml(QUT_B / "calibration.xml")
    roi = [[330, 0], [703, 0], [703, 575], [330, 575]]  # 자료에 든 roi.xml
    assert image_polygon_ground_area(cam, roi) == pytest.approx(52.5, abs=0.5)  # E-30
    assert cam.position[2] == pytest.approx(6.6, abs=0.1)
