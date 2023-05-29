#!/usr/bin/env python3

import time
from os import PathLike

import mouse
import pyarrow as pa
from pyarrow.parquet import ParquetWriter

schema = pa.schema(
    [
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("time", pa.float32()),
    ]
)

last_x, last_y, last_time = None, None, None


def mouse_callback(
    event: mouse.ButtonEvent | mouse.WheelEvent | mouse.MoveEvent, writer: ParquetWriter
):
    global last_x, last_y, last_time

    if not isinstance(event, mouse.MoveEvent):
        return

    if last_x is not None and (last_x != event.x or last_y != event.y):
        data = {
            "x": pa.array([event.x - last_x], type=pa.float32()),
            "y": pa.array([event.y - last_y], type=pa.float32()),
            "time": pa.array([event.time - last_time], type=pa.float32()),
        }

        writer.write(pa.table(data))

    last_x, last_y, last_time = event.x, event.y, event.time


def record(cursor_dataset: PathLike | str):
    writer = ParquetWriter(cursor_dataset, schema)
    mouse.hook(lambda event: mouse_callback(event, writer))

    try:
        while True:
            time.sleep(1000)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
