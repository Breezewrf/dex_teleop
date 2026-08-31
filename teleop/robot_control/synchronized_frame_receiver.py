#!/usr/bin/env python3
"""Test and Inspect synchronized teleop frames without controlling a robot."""

import argparse
import json
import logging

import zmq


LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subscribe to atomic arm/left-hand/right-hand teleop frames."
    )
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:8560")
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print complete JSON frames instead of one-line summaries.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Exit after this many frames; 0 runs until Ctrl+C.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(args.endpoint)
    LOGGER.info("Subscribed to %s", args.endpoint)

    received = 0
    last_frame_id = None
    try:
        while args.count == 0 or received < args.count:
            try:
                frame = socket.recv_json()
            except zmq.Again:
                LOGGER.warning("No synchronized frame received for %d ms", args.timeout_ms)
                continue

            frame_id = frame.get("frame_id")
            if last_frame_id is not None and frame_id != last_frame_id + 1:
                LOGGER.warning(
                    "Frame gap: expected %d, received %s",
                    last_frame_id + 1,
                    frame_id,
                )
            last_frame_id = frame_id
            received += 1

            if args.print_json:
                print(json.dumps(frame, ensure_ascii=False))
            else:
                print(
                    "frame={frame_id} mode={mode} arm={arm} left={left} "
                    "right={right} timestamp_ns={timestamp_ns}".format(
                        frame_id=frame_id,
                        mode=frame.get("mode"),
                        arm=frame.get("arm", {}).get("valid"),
                        left=frame.get("left_hand", {}).get("valid"),
                        right=frame.get("right_hand", {}).get("valid"),
                        timestamp_ns=frame.get("timestamp_ns"),
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
