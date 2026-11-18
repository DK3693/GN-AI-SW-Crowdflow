# 이 파일이 하는 일: 통과 시험(4단계). 영상에서 사람을 찾아 구역별로 세고, 정답 라벨과 비교하고,
#   평균 시간에 따라 숫자가 얼마나 흔들리는지 잰다. 결과를 outputs/pass_test/ 에 남긴다.
# 해당 기능: 핵심 기능 1 (사람 세기)
# 해당 기준: C-08, C-23, C-30, C-37, C-20
# 근거: D-15 (통과 시험), E-36, E-37, E-44 (자료), Q-12, Q-13
# 출처: AI 코딩 도구로 작성
#
# 실행: PYTHONPATH=src uv run python -m crowdflow.pass_test --data data/aihub489_sample/ja-ma
from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from crowdflow.crowd_vision import count_by_zone, load_yaml, load_zones, moving_average

ROOT = Path(__file__).resolve().parents[2]


def read_truth(json_path: Path, zones) -> tuple[dict[int, dict[str, int]], int, float]:
    """AI Hub 489 라벨에서 프레임별·구역별 정답 사람 수를 만든다."""
    d = json.loads(json_path.read_text(encoding="utf-8"))
    boxes = collections.defaultdict(list)
    for a in d["annotations"]:
        boxes[a["frame"]].append(a["bbox"])
    n = d["video"]["total_frame"]
    truth = {f: count_by_zone(np.array(boxes.get(f, [])), zones) for f in range(n)}
    return truth, n, float(d["video"]["fps"])


def run_model(model: YOLO, video: Path, zones, det: dict, batch: int = 16):
    """영상 한 개를 처음부터 끝까지 읽어 프레임별·구역별 사람 수를 돌려준다."""
    cap = cv2.VideoCapture(str(video))
    counts, frames = [], []

    def flush():
        results = model.predict(
            frames, classes=[det["person_class_id"]], conf=det["conf"], iou=det["iou"],
            imgsz=det["imgsz"], verbose=False,
        )
        for r in results:
            counts.append(count_by_zone(r.boxes.xyxy.cpu().numpy(), zones))
        frames.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) == batch:
            flush()
    if frames:
        flush()
    cap.release()
    return counts


def save_example(model: YOLO, video: Path, frame_no: int, zone_cfg: dict, det: dict, out: Path):
    """한 장면에 구역 선과 찾은 사람 상자를 그려 저장한다 (눈으로 확인하는 용도)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    r = model.predict(img, classes=[det["person_class_id"]], conf=det["conf"], iou=det["iou"],
                      imgsz=det["imgsz"], verbose=False)[0]
    for name, z in zone_cfg["zones"].items():
        pts = np.array(z["polygon"], dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 200, 0), 3)
        cv2.putText(img, name, (pts[0][0] + 15, pts[0][1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 200, 0), 3)
    for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img, ((x1 + x2) // 2, y2), 6, (0, 0, 255), -1)
    cv2.imencode(".jpg", img)[1].tofile(str(out))  # 한글 경로에서도 저장되게 함


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="json/ 과 video/ 폴더가 든 시간대 폴더들의 위 폴더")
    p.add_argument("--zones", default="config/zones/aihub489_ja-ma_C0041.yaml")
    p.add_argument("--models", nargs="*", help="비우면 settings.yaml의 후보를 모두 돌린다")
    p.add_argument("--imgsz", type=int, help="비우면 settings.yaml의 값을 쓴다")
    p.add_argument("--limit", type=int, help="영상 개수 제한 (빠른 확인용)")
    p.add_argument("--tag", default="run")
    args = p.parse_args()

    settings = load_yaml(ROOT / "config" / "settings.yaml")
    det = dict(settings["detection"])
    det["imgsz"] = args.imgsz or settings["pass_test"]["imgsz"]  # 통과 시험의 기본 입력 크기는 따로 둔다
    models = args.models or settings["pass_test"]["model_candidates"]
    windows_s = settings["pass_test"]["smoothing_seconds_candidates"]
    zone_cfg = load_yaml(ROOT / args.zones)
    zones = load_zones(ROOT / args.zones)
    names = list(zones)

    jsons = sorted((ROOT / args.data).glob("*/json/*.json"))[: args.limit]
    out_dir = ROOT / "outputs" / "pass_test" / f"{args.tag}_imgsz{det['imgsz']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    truths = {}
    for j in jsons:
        truths[j] = read_truth(j, zones)

    summary, smooth_rows, frame_rows = [], [], []
    for mname in models:
        model = YOLO(mname)
        t0, n_frames = time.time(), 0
        pred_all, true_all = [], []  # 영상별 (프레임 × 구역) 배열
        for j in jsons:
            truth, n, fps = truths[j]
            video = j.parent.parent / "video" / (j.stem + ".mp4")
            counts = run_model(model, video, zones, det)
            m = min(n, len(counts))
            pred = np.array([[counts[f][z] for z in names] for f in range(m)])
            true = np.array([[truth[f][z] for z in names] for f in range(m)])
            pred_all.append(pred)
            true_all.append(true)
            n_frames += m
            for f in range(m):
                frame_rows.append([mname, j.stem, f, *pred[f], *true[f]])
        secs = time.time() - t0
        P, T = np.vstack(pred_all), np.vstack(true_all)
        row = {
            "model": mname, "imgsz": det["imgsz"], "clips": len(jsons), "frames": n_frames,
            "fps_processed": round(n_frames / secs, 1),
            "true_mean_total": round(T.sum(1).mean(), 2), "pred_mean_total": round(P.sum(1).mean(), 2),
            "mae_total": round(np.abs(P.sum(1) - T.sum(1)).mean(), 3),
            "bias_total": round((P.sum(1) - T.sum(1)).mean(), 3),
            "exact_total_pct": round((P.sum(1) == T.sum(1)).mean() * 100, 1),
            "within1_total_pct": round((np.abs(P.sum(1) - T.sum(1)) <= 1).mean() * 100, 1),
        }
        for i, z in enumerate(names):
            row[f"mae_{z}"] = round(np.abs(P[:, i] - T[:, i]).mean(), 3)
            row[f"bias_{z}"] = round((P[:, i] - T[:, i]).mean(), 3)
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        summary.append(row)
        print(row, flush=True)

        # 평균 시간별 흔들림: 구역별 값이 한 프레임 사이에 평균 얼마나 바뀌는가, 정답과의 차이는 얼마인가
        fps = truths[jsons[0]][2]
        for w_s in [0] + windows_s:
            w = max(1, int(round(w_s * fps)))
            jit, mae, jit_true = [], [], []
            for pred, true in zip(pred_all, true_all):
                for i in range(len(names)):
                    sp = moving_average(pred[:, i], w)
                    st = moving_average(true[:, i], w)
                    jit.append(np.abs(np.diff(sp)).mean())
                    jit_true.append(np.abs(np.diff(st)).mean())
                    mae.append(np.abs(sp - true[:, i]).mean())
            smooth_rows.append({
                "model": mname, "window_s": w_s, "window_frames": w,
                "jitter_pred": round(float(np.mean(jit)), 4),
                "jitter_truth_same_window": round(float(np.mean(jit_true)), 4),
                "mae_vs_raw_truth": round(float(np.mean(mae)), 4),
            })

        # 눈으로 볼 장면 하나: 첫 영상에서 정답 사람 수가 가장 많은 프레임
        j0 = jsons[0]
        busiest = int(true_all[0].sum(1).argmax())
        save_example(model, j0.parent.parent / "video" / (j0.stem + ".mp4"), busiest, zone_cfg, det,
                     out_dir / f"example_{Path(mname).stem}.jpg")

    def write(name, rows):
        with open(out_dir / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("summary.csv", summary)
    write("smoothing.csv", smooth_rows)
    with open(out_dir / "per_frame.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "clip", "frame", *[f"pred_{z}" for z in names], *[f"true_{z}" for z in names]])
        w.writerows(frame_rows)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
