"""動画フレームの読み出し共通処理。

中継映像にはインターレース素材由来のフレーム(top_field_first等のフラグ付き)が
混在しており、opencv-pythonに内蔵されたFFmpegではYUV->BGR変換に失敗し、
エラーは出さずに全フレームが真っ黒になる(swscalerの
"Cannot convert interlaced to progressive frames or vice versa" が原因)。
PyAV(av)は同じ映像を正しくデコードできるため、フレーム読み出しはすべて
PyAV経由で行う。
"""

from __future__ import annotations

import av


def open_video(path: str) -> av.container.InputContainer:
    """PyAVでコンテナを開く。使い終わったらclose()すること。"""
    return av.open(path)


def get_video_meta(path: str) -> dict:
    container = open_video(path)
    try:
        return stream_meta(container.streams.video[0])
    finally:
        container.close()


def stream_meta(stream) -> dict:
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    total = stream.frames
    if not total and stream.duration:
        total = int(round(float(stream.duration * stream.time_base) * fps))
    return {"frames": total or 0, "fps": fps, "width": stream.width, "height": stream.height}


def read_frame_bgr(container: av.container.InputContainer, frame_num: int):
    """container: open_video()で開いたコンテナ。指定フレーム(BGR ndarray)を返す。失敗時はNone。

    可変フレームレート素材ではフレーム番号どおりの厳密なシークができないことがあるため、
    pts基準で目的フレーム以降の最初のフレームを返し、末尾を超える場合は
    デコードできた最後のフレームにフォールバックする。
    """
    stream = container.streams.video[0]
    total = stream.frames or 0
    frame_num = max(0, min(frame_num, total - 1)) if total > 0 else max(0, frame_num)

    fps = stream.average_rate
    time_base = stream.time_base
    target_pts = int(frame_num / fps / time_base) if fps and time_base else 0

    try:
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
    except av.error.FFmpegError:
        container.seek(0, stream=stream)

    last = None
    for frame in container.decode(stream):
        last = frame.to_ndarray(format="bgr24")
        if frame.pts is None or frame.pts >= target_pts:
            return last
    return last
