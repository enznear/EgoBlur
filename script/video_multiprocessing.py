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


def _process_segment(args: tuple[str, int, int, str]) -> str:
    """Process a segment of the video.

    Parameters
    ----------
    args: tuple containing
        input_video_path: str
        start_frame: int
        end_frame: int
        output_path: str

    Returns
    -------
    str
        Path to the processed segment file.
    """

    input_video_path, start_frame, end_frame, output_path = args

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        processed = cv2.GaussianBlur(frame, (21, 21), 0)
        writer.write(processed)

    writer.release()
    cap.release()
    return output_path


def process_video_multiprocessing(
    input_video_path: str,
    output_video_path: str,
    num_processes: int,
) -> None:
    """Split ``input_video_path`` into ``num_processes`` chunks and process them
    in parallel. The processed chunks are concatenated and written to
    ``output_video_path``.
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
        segments.append((input_video_path, start, end, part_path))
        start = end
        idx += 1

    with Pool(processes=num_processes) as pool:
        part_paths = pool.map(_process_segment, segments)

    clips = [VideoFileClip(p) for p in part_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_video_path, fps=fps)
    final_clip.close()
    for clip in clips:
        clip.close()
        os.remove(clip.filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process video using multiprocessing")
    parser.add_argument("--input_video_path", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    parser.add_argument("--num_processes", default=2, type=int, help="Number of worker processes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video_multiprocessing(
        args.input_video_path,
        args.output_video_path,
        max(1, args.num_processes),
    )
