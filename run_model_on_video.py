from __future__ import annotations

import argparse
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from run_model import predict_single
from sr_project.image_utils import save_rgb_image
from sr_project.models import build_model
from sr_project.runtime import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video super-resolution: frames → upscale (same as run_model.py) → video."
    )
    parser.add_argument("--model", choices=["srcnn", "fsrcnn"], required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4, 8])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("artifacts/predictions/video_output.mp4"),
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _probe_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = result.stdout.strip()
    return float(Fraction(s)) if "/" in s else float(s)


def _extract_frames(video: Path, frames_dir: Path) -> float:
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            str(frames_dir / "frame_%06d.png"),
        ]
    )
    return _probe_fps(video)


def _frames_to_video(frames_dir: Path, out: Path, fps: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    codec = (
        "libx264"
        if encoders.returncode == 0 and "libx264" in encoders.stdout
        else "mpeg4"
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
    )


def main() -> None:
    args = parse_args()

    if not args.input_video.exists():
        raise SystemExit(f"Missing input video: {args.input_video}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    device = resolve_device("auto")
    model = build_model(args.model, args.scale).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    with tempfile.TemporaryDirectory(prefix="sr_video_") as tmp:
        tmp_path = Path(tmp)
        frames_in = tmp_path / "in"
        frames_out = tmp_path / "out"
        frames_out.mkdir(parents=True, exist_ok=True)

        print(f"Extracting frames from {args.input_video}...")
        fps = _extract_frames(args.input_video, frames_in)

        paths = sorted(frames_in.glob("frame_*.png"))
        if not paths:
            raise RuntimeError(f"No frames extracted to {frames_in}")

        print(f"Upscaling {len(paths)} frames ({args.model}, x{args.scale})...")
        for i, frame_path in enumerate(paths, start=1):
            output_rgb, _ = predict_single(args.model, model, frame_path, args.scale, device)
            save_rgb_image(output_rgb, frames_out / f"frame_{i:06d}.png")

        print(f"Writing {args.output_video}...")
        _frames_to_video(frames_out, args.output_video, fps)

    print(f"Done: {args.output_video}")


if __name__ == "__main__":
    main()
