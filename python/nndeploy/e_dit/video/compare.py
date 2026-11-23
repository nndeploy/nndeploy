# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import closing
from itertools import islice, zip_longest
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class FrameMetrics:
    index: int
    ssim: float
    mse: float
    mae: float
    psnr: float
    max_abs_diff: int
    pass_frame: bool


def _read_video(path: str):
    """Decode one frame at a time and close the decoder on all exit paths."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not count:
            raise ValueError(f"No frames in video: {path}")
    finally:
        cap.release()


def _ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(score.mean())


def compare_videos(
    reference: str,
    candidate: str,
    *,
    min_ssim: float = 0.90,
    max_mse: float = 150.0,
    max_mae: float = 8.0,
    allow_frame_count_delta: int = 1,
    max_failed_frame_ratio: float = 0.05,
    drop_reference_first_frame: bool = False,
    drop_candidate_first_frame: bool = False,
) -> dict:
    if not 0 <= min_ssim <= 1 or not 0 <= max_failed_frame_ratio <= 1:
        raise ValueError('SSIM and failed-frame ratio thresholds must be in [0, 1]')
    if any(not math.isfinite(v) or v < 0 for v in (max_mse, max_mae)):
        raise ValueError('Error thresholds must be finite and nonnegative')
    if type(allow_frame_count_delta) is not int or allow_frame_count_delta < 0:
        raise ValueError('allow_frame_count_delta must be a nonnegative integer')
    frame_reports: list[FrameMetrics] = []
    ref_count = cand_count = 0
    with closing(_read_video(reference)) as ref_reader, closing(_read_video(candidate)) as cand_reader:
        ref_frames = islice(ref_reader, int(drop_reference_first_frame), None)
        cand_frames = islice(cand_reader, int(drop_candidate_first_frame), None)
        for idx, (ref, cand) in enumerate(zip_longest(ref_frames, cand_frames)):
            ref_count += ref is not None
            cand_count += cand is not None
            if ref is None or cand is None:
                continue
            if ref.shape != cand.shape:
                cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]))
            diff = ref.astype(np.float32) - cand.astype(np.float32)
            mse = float(np.mean(diff * diff))
            mae = float(np.mean(np.abs(diff)))
            max_abs = int(np.max(np.abs(diff)))
            psnr = float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))
            ssim = _ssim(ref, cand)
            pass_frame = ssim >= min_ssim and mse <= max_mse and mae <= max_mae
            frame_reports.append(FrameMetrics(idx, ssim, mse, mae, psnr, max_abs, pass_frame))

    if abs(ref_count - cand_count) > allow_frame_count_delta:
        raise ValueError(f'Frame count mismatch: reference={ref_count}, candidate={cand_count}, allowed_delta={allow_frame_count_delta}')
    compared = len(frame_reports)
    if not compared:
        raise ValueError('No comparable frames')

    failed = [m.index for m in frame_reports if not m.pass_frame]
    failed_ratio = len(failed) / compared
    summary = {
        "compared_frames": compared,
        "ssim_mean": float(np.mean([m.ssim for m in frame_reports])),
        "ssim_min": float(np.min([m.ssim for m in frame_reports])),
        "mse_mean": float(np.mean([m.mse for m in frame_reports])),
        "mse_max": float(np.max([m.mse for m in frame_reports])),
        "mae_mean": float(np.mean([m.mae for m in frame_reports])),
        "mae_max": float(np.max([m.mae for m in frame_reports])),
        "psnr_mean": float(np.mean([m.psnr for m in frame_reports if math.isfinite(m.psnr)]))
        if any(math.isfinite(m.psnr) for m in frame_reports)
        else float("inf"),
        "max_abs_diff": int(np.max([m.max_abs_diff for m in frame_reports])),
        "failed_frames": failed,
        "pass_compare": failed_ratio <= max_failed_frame_ratio,
        "thresholds": {
            "min_ssim": min_ssim,
            "max_mse": max_mse,
            "max_mae": max_mae,
            "allow_frame_count_delta": allow_frame_count_delta,
            "max_failed_frame_ratio": max_failed_frame_ratio,
        },
    }
    return {"summary": summary, "frames": [asdict(m) for m in frame_reports]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two videos frame-by-frame.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--min-ssim", type=float, default=0.90)
    parser.add_argument("--max-mse", type=float, default=150.0)
    parser.add_argument("--max-mae", type=float, default=8.0)
    parser.add_argument("--allow-frame-count-delta", type=int, default=1)
    parser.add_argument("--max-failed-frame-ratio", type=float, default=0.05)
    parser.add_argument("--drop-reference-first-frame", action="store_true")
    parser.add_argument("--drop-candidate-first-frame", action="store_true")
    args = parser.parse_args()
    report = compare_videos(
        args.reference,
        args.candidate,
        min_ssim=args.min_ssim,
        max_mse=args.max_mse,
        max_mae=args.max_mae,
        allow_frame_count_delta=args.allow_frame_count_delta,
        max_failed_frame_ratio=args.max_failed_frame_ratio,
        drop_reference_first_frame=args.drop_reference_first_frame,
        drop_candidate_first_frame=args.drop_candidate_first_frame,
    )
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
