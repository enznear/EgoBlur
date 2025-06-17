# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Example script that splits a video into segments and processes them
in parallel using Python multiprocessing."""

import argparse
import math
import os
from multiprocessing import Pool

import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch

from script.demo_ego_blur import _process_frame, get_device


def _process_segment(
    args: tuple[
        str,
        int,
        int,
        str,
        str | None,
        str | None,
        float,
        float,
        float,
        float,
    ]
) -> str:

    """Process a segment of the video.

    Parameters
    ----------
    args: tuple containing
        input_video_path: str
        start_frame: int
        end_frame: int
        output_path: str
        face_model_path: str | None
        lp_model_path: str | None
        face_model_score_threshold: float
        lp_model_score_threshold: float
        nms_iou_threshold: float
        scale_factor_detections: float


    Returns
    -------
    str
        Path to the processed segment file.
    """

    (
        input_video_path,
        start_frame,
        end_frame,
        output_path,
        face_model_path,
        lp_model_path,
        face_model_score_threshold,
        lp_model_score_threshold,
        nms_iou_threshold,
        scale_factor_detections,
    ) = args

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if face_model_path is not None:
        face_detector = torch.jit.load(face_model_path, map_location="cpu").to(
            get_device()
        )
        face_detector.eval()
    else:
        face_detector = None

    if lp_model_path is not None:
        lp_detector = torch.jit.load(lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

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
        Path to the face detection model.
    lp_model_path: str | None
        Path to the license plate detection model.
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

    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

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
                face_model_path,
                lp_model_path,
                face_model_score_threshold,
                lp_model_score_threshold,
                nms_iou_threshold,
                scale_factor_detections,
            )
        )

        start = end
        idx += 1

    with Pool(processes=num_processes) as pool:
        part_paths = pool.map(_process_segment, segments)


    output_fps = output_video_fps if output_video_fps is not None else fps
    clips = [VideoFileClip(p) for p in part_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_video_path, fps=output_fps)
    final_clip.close()
    for clip in clips:
        clip.close()
        os.remove(clip.filename)


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
