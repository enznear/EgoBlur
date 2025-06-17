# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Example script that splits a video into segments and processes them
in parallel using Python multiprocessing."""

import argparse
import math
import time
import os
from multiprocessing import Manager
import torch.multiprocessing as mp

import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch

from script.demo_ego_blur import _process_frame, get_device



# Global detectors that are initialized once and shared across workers

FACE_DETECTOR = None
LP_DETECTOR = None


def init_worker(fd, lp):
    """Set global detectors for each worker.

    ``fd`` and ``lp`` can be either loaded models (when using ``fork``)
    or file paths (when using ``spawn``).
    """
    global FACE_DETECTOR, LP_DETECTOR

    if isinstance(fd, str) or fd is None:
        FACE_DETECTOR = (
            torch.jit.load(fd, map_location="cpu").to(get_device()).eval()
            if fd is not None
            else None
        )
    else:
        FACE_DETECTOR = fd

    if isinstance(lp, str) or lp is None:
        LP_DETECTOR = (
            torch.jit.load(lp, map_location="cpu").to(get_device()).eval()
            if lp is not None
            else None
        )
    else:
        LP_DETECTOR = lp



def print_progress(
    iteration: int,
    total: int,
    prefix: str = "",
    length: int = 30,
    ratio: float | None = None,
) -> None:
    """Simple progress bar with optional realtime ratio."""

    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    ratio_text = f" ({ratio:.2f}x)" if ratio is not None else ""
    print(f"\r{prefix} |{bar}| {percent:.1f}%{ratio_text}", end="")
    if iteration >= total:
        print()



def _process_segment(
    input_video_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
    progress_queue=None,
    progress_step: int = 1,
) -> str:

    """Process a segment of the video.

    Parameters
    ----------
    input_video_path: str
        Path to the input video.
    start_frame: int
        First frame of the segment to process.
    end_frame: int
        Last frame (exclusive) of the segment to process.
    output_path: str
        Path where the processed segment will be written.
    face_model_score_threshold: float
        Threshold for filtering face detections.
    lp_model_score_threshold: float
        Threshold for filtering license plate detections.
    nms_iou_threshold: float
        NMS IoU threshold to filter overlapping boxes.
    scale_factor_detections: float
        Scale factor for detection boxes.
    progress_queue: multiprocessing.Queue | None
        If provided, will be used to send progress updates.
    progress_step: int
        Number of frames processed before sending a progress update.


    Returns
    -------
    str
        Path to the processed segment file.
    """

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_detector = FACE_DETECTOR
    lp_detector = LP_DETECTOR

    count = 0

    for _ in range(start_frame, end_frame):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        processed = _process_frame(
            frame_rgb,
            face_detector,
            lp_detector,
            face_model_score_threshold,
            lp_model_score_threshold,
            nms_iou_threshold,
            scale_factor_detections,
        )
        writer.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        count += 1
        if progress_queue is not None and count % progress_step == 0:
            progress_queue.put(progress_step)

    if progress_queue is not None and count % progress_step != 0:
        progress_queue.put(count % progress_step)

    writer.release()
    cap.release()
    return output_path


def process_video_multiprocessing(
    input_video_path: str,
    output_video_path: str,
    num_processes: int,
    face_model_path: str | None,
    lp_model_path: str | None,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
    output_video_fps: int | None,
) -> None:
    """Split ``input_video_path`` into ``num_processes`` chunks and process them
    in parallel. The processed chunks are concatenated and written to
    ``output_video_path``.

    Parameters
    ----------
    input_video_path: str
        Path to the input video.
    output_video_path: str
        Where to save the processed video.
    num_processes: int
        Number of worker processes to use.
    face_model_path: str | None
        Path to the face detection model. Loaded once and shared across workers.
    lp_model_path: str | None
        Path to the license plate detection model. Loaded once and shared across workers.

    face_model_score_threshold: float
        Threshold for filtering face detections.
    lp_model_score_threshold: float
        Threshold for filtering license plate detections.
    nms_iou_threshold: float
        NMS IoU threshold to filter overlapping boxes.
    scale_factor_detections: float
        Scale factor for detection boxes.
    output_video_fps: int | None
        FPS of the output video. Defaults to the input FPS if ``None``.
    """

    start_method = "spawn" if torch.cuda.is_available() else "fork"

    face_detector = None
    lp_detector = None
    init_args = (face_model_path, lp_model_path)

    if start_method == "fork":
        if face_model_path is not None:
            face_detector = torch.jit.load(face_model_path, map_location="cpu").to(
                get_device()
            )
            face_detector.eval()
            face_detector.share_memory()

        if lp_model_path is not None:
            lp_detector = torch.jit.load(lp_model_path, map_location="cpu").to(
                get_device()
            )
            lp_detector.eval()
            lp_detector.share_memory()

        init_args = (face_detector, lp_detector)


    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps else 0
    cap.release()

    progress_step = max(1, total_frames // 100)

    frames_per_chunk = math.ceil(total_frames / max(1, num_processes))

    base, ext = os.path.splitext(output_video_path)
    segments = []
    start = 0
    idx = 0
    while start < total_frames:
        end = min(start + frames_per_chunk, total_frames)
        part_path = f"{base}_part{idx}{ext}"
        segments.append(
            (
                input_video_path,
                start,
                end,
                part_path,
                face_model_score_threshold,
                lp_model_score_threshold,
                nms_iou_threshold,
                scale_factor_detections,
            )
        )
        start = end
        idx += 1

    with Manager() as manager:
        progress_queue = manager.Queue()
        start_time = time.time()
        processed = 0

        ctx = mp.get_context(start_method)
        with ctx.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:

            results = [
                pool.apply_async(
                    _process_segment,
                    args=(*seg, progress_queue, progress_step),
                )
                for seg in segments
            ]

            while processed < total_frames:
                processed += progress_queue.get()
                elapsed = time.time() - start_time
                video_time = processed / fps if fps else 0
                ratio_now = elapsed / video_time if video_time else 0
                print_progress(
                    processed,
                    total_frames,
                    prefix="Processing",
                    ratio=ratio_now,
                )

            part_paths = [r.get() for r in results]

    output_fps = output_video_fps if output_video_fps is not None else fps
    clips = [VideoFileClip(p) for p in part_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_video_path, fps=output_fps)
    final_clip.close()
    for clip in clips:
        clip.close()
        os.remove(clip.filename)

    elapsed_time = time.time() - start_time
    ratio = elapsed_time / video_duration if video_duration else 0
    print(
        f"Video processing completed in {elapsed_time:.2f} seconds. "
        f"({ratio:.2f}x realtime)"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process video using multiprocessing with EgoBlur parameters"
    )
    parser.add_argument("--input_video_path", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    parser.add_argument(
        "--num_processes",
        default=2,
        type=int,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur face model file path",
    )
    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help="Face model score threshold to filter out low confidence detections",
    )
    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur license plate model file path",
    )
    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.9,
        help=(
            "License plate model score threshold to filter out low confidence detections"
        ),
    )
    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.3,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )
    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help=(
            "Scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling"
        ),
    )
    parser.add_argument(
        "--output_video_fps",
        required=False,
        type=int,
        default=None,
        help="FPS for the output video. If not provided, the input video's FPS is used",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video_multiprocessing(
        args.input_video_path,
        args.output_video_path,
        max(1, args.num_processes),
        args.face_model_path,
        args.lp_model_path,
        args.face_model_score_threshold,
        args.lp_model_score_threshold,
        args.nms_iou_threshold,
        args.scale_factor_detections,
        args.output_video_fps,
    )
