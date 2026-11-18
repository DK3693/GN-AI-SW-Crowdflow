# 이 파일이 하는 일: 영상(동영상 파일 또는 사진 묶음)의 장마다 사람을 찾아, 상자를 파일 하나에 저장한다.
#   구역을 바꿔 가며 다시 셀 때 영상 분석을 처음부터 다시 하지 않기 위한 것이다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-23, C-30, C-37
# 근거: E-27 (통과 시험에서 쓴 방법과 설정), E-36, E-45, D-26 (학습 없이 추론만), D-44 (동영상 파일을 주 자료로)
# 출처: AI 코딩 도구로 작성
#
# 실행 (동영상): PYTHONPATH=src uv run python -m crowdflow.detect_cache --video <파일.mp4> --step 3 --model yolo26x.pt --imgsz 1280 --out outputs/detections/x.npz
# 실행 (사진 묶음): PYTHONPATH=src uv run python -m crowdflow.detect_cache --frames <폴더> --model yolo26x.pt --imgsz 1280 --out outputs/detections/x.npz
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from crowdflow.crowd_vision import load_yaml

ROOT = Path(__file__).resolve().parents[2]


def imread(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # 한글 경로에서도 읽힘


def iter_folder(folder: Path):
    """사진 묶음: (프레임 번호, 사진)을 차례로 낸다. 프레임 번호는 파일 이름이다."""
    for f in sorted(folder.glob("*.jpg")):
        yield int(f.stem), imread(f)


def iter_video(path: Path, step: int):
    """동영상: step장마다 한 장씩 (원래 프레임 번호, 사진)을 낸다. 건너뛰지 않고 차례로 읽는다(건너뛰기는 느리고 어긋날 수 있음)."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, frame
        i += 1
    cap.release()


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="사진(.jpg)이 든 폴더")
    src.add_argument("--video", help="동영상 파일")
    p.add_argument("--step", type=int, default=1, help="동영상에서 몇 장마다 한 장을 볼지")
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    det = load_yaml(ROOT / "config" / "settings.yaml")["detection"]
    if args.video:
        cap = cv2.VideoCapture(str(ROOT / args.video))
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        source = iter_video(ROOT / args.video, args.step)
    else:
        fps, total = float("nan"), len(list((ROOT / args.frames).glob("*.jpg")))  # 사진 묶음에는 초당 장수가 적혀 있지 않다
        source = iter_folder(ROOT / args.frames)

    model = YOLO(args.model)
    seen, frame_idx, boxes, confs = [], [], [], []
    t0 = time.time()

    def flush(batch):
        results = model.predict([f for _, f in batch], classes=[det["person_class_id"]], conf=det["conf"],
                                iou=det["iou"], imgsz=args.imgsz, verbose=False)
        for (i, _), r in zip(batch, results):
            b = r.boxes.xyxy.cpu().numpy()
            seen.append(i)
            frame_idx.append(np.full(len(b), i))
            boxes.append(b)
            confs.append(r.boxes.conf.cpu().numpy())

    batch = []
    for item in source:
        batch.append(item)
        if len(batch) == args.batch:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, frame=np.concatenate(frame_idx), xyxy=np.vstack(boxes), conf=np.concatenate(confs),
        frames_seen=np.array(seen), n_frames=total, fps=fps, step=args.step,
        model=args.model, imgsz=args.imgsz, conf_threshold=det["conf"], iou=det["iou"],
    )
    print(f"frames analysed {len(seen)} of {total}, detections {sum(len(b) for b in boxes)}, {len(seen) / (time.time() - t0):.1f} fps -> {out}")


if __name__ == "__main__":
    main()
