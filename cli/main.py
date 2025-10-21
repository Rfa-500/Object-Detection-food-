from __future__ import annotations

import argparse
import json
from typing import Optional

import cv2
from tqdm import tqdm

from core.bot import FoodDetectionBot
from core.config_loader import AppConfig, load_config
from utils.logging_utils import configure_logging, get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food & Snack Detection Bot")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--camera", type=int, help="Camera index for real-time detection")
    parser.add_argument("--out", type=str, help="Path to output JSON file")
    parser.add_argument("--export-csv", action="store_true", help="Export results to CSV (requires --out)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def load_bot(config_path: str) -> FoodDetectionBot:
    config = load_config(config_path)
    return FoodDetectionBot(config)


def run_image(bot: FoodDetectionBot, image_path: str, output_path: Optional[str], export_csv: bool) -> None:
    stages = ["loading_image", "running_detection", "estimating_depth", "estimating_portions", "completed"]
    progress = tqdm(total=len(stages), desc="Processing")

    def progress_cb(stage: str) -> None:
        if stage in stages:
            progress.update(1)
            progress.set_postfix(stage=stage)

    try:
        results = bot.process_image(image_path, export_csv=export_csv, output_path=output_path, progress_cb=progress_cb)
    except Exception as exc:
        progress.close()
        logger.error("Failed to process image: %s", exc)
        raise
    progress.close()

    summary = {
        "image": image_path,
        "items_detected": len(results.get("detections", [])),
        "calories": results.get("totals", {}).get("calories", 0.0),
        "protein_g": results.get("totals", {}).get("protein_g", 0.0),
        "fat_g": results.get("totals", {}).get("fat_g", 0.0),
        "sugar_g": results.get("totals", {}).get("sugar_g", 0.0),
        "sodium_mg": results.get("totals", {}).get("sodium_mg", 0.0),
        "overall_reliability": results.get("overall_reliability", 0.0),
    }
    logger.info("Summary: %s", json.dumps(summary, indent=2))

    if not output_path:
        print(json.dumps(results, indent=2))


def run_camera(bot: FoodDetectionBot, camera_index: int) -> None:
    runtime_cfg = bot.runtime_opts
    frame_skip = int(runtime_cfg.get("frame_skip", 2))
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    frame_idx = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % max(frame_skip, 1) != 0:
                continue
            cv2.imshow("Food Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    configure_logging(level=getattr(__import__("logging"), args.log_level.upper(), None) or 20)
    bot = load_bot(args.config)

    if args.image:
        run_image(bot, args.image, args.out, args.export_csv)
    elif args.camera is not None:
        run_camera(bot, args.camera)
    else:
        raise SystemExit("Either --image or --camera must be provided")


if __name__ == "__main__":
    main()
