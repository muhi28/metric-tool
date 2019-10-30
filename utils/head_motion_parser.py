import argparse
import csv
import sys

from utils.vector_util import Vector3


def _parse_raw_lines(lines):
    raw_lines = []

    # parse raw data lines
    for hm in lines:

        if "VD-FRAME" not in hm[0]:
            continue

        # convert raw log to frame log
        _, time_stamp, x_val = hm[0].split(": ")

        vec3 = Vector3()
        vec3.x = float(x_val)
        vec3.y = float(hm[1])
        vec3.z = float(hm[2])

        raw_lines.append((int(time_stamp), vec3))

    print(f"raw count lines {len(raw_lines)}")
    return raw_lines


def process_log(log_path, fps, num_frames):
    """
        Parse head movement data from log file

    :param log_path: path to log file
    :param fps: frames per second
    :param num_frames: number of video frames
    :return:
    """

    frame_duration = 1000.0 / fps
    result = []

    with open(log_path, "r") as log_file:
        lines = csv.reader(log_file, delimiter=",")

        head_mvmts = _parse_raw_lines(lines)

        # convert to frame log list
        next_frame_time = 0
        first_entry_time, _ = head_mvmts[0]

        for entry_time, view_direction in head_mvmts:

            offset = entry_time - first_entry_time

            if offset >= next_frame_time:
                result.append(view_direction)
                next_frame_time += frame_duration

            if len(result) >= num_frames:
                break

    return result
