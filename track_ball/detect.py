import os
import queue
import sys
import threading
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from models.tracknet import TrackNet
from utils.general import get_shuttle_position

# from yolov5 detect.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "example_dataset/match/videos/1_10_12.mp4",
        help="Path to video.",
    )
    parser.add_argument("--view-img", action="store_true", help="save frame images")
    parser.add_argument("--save-video", action="store_true", help="save annotated video (default: off)")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[288, 512],
        help="image size h,w",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=ROOT / "best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 half-precision inference (default: off)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile model optimization",
    )
    opt = parser.parse_args()
    return opt


SQ = 3  # frames per model input sample


def _video_writer_worker(q: queue.Queue, out: cv2.VideoWriter):
    while True:
        frame = q.get()
        if frame is None:
            break
        out.write(frame)


def main(opt):
    source_name = os.path.splitext(os.path.basename(opt.source))[0]
    b_view_img = opt.view_img
    b_save_video = opt.save_video
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    f_source = str(opt.source)
    imgsz = opt.imgsz
    use_fp16 = opt.fp16
    use_compile = not opt.no_compile

    #source_name = "{}_predict".format(source_name)
    source_name = "{}".format(source_name)

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)

    img_save_path = "{}/{}".format(d_save_dir, source_name)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(f_weights, weights_only=True))
    model.eval()

    if use_fp16 and device == "cuda":
        model = model.half()

    if use_compile:
        print("Compiling model with torch.compile...")
        t0 = time.perf_counter()
        model = torch.compile(model)
        dummy = torch.zeros(1, SQ * 3, imgsz[0], imgsz[1], dtype=dtype, device=device)
        with torch.inference_mode():
            _ = model(dummy)
        del dummy
        print("Warm-up done. ({:.1f}s)".format(time.perf_counter() - t0))

    # 固定サイズの GPU バッファを事前確保（毎イテレーションの malloc/free を回避）
    input_buffer = torch.empty((1, SQ * 3, imgsz[0], imgsz[1]), device=device, dtype=dtype)

    vid_cap = cv2.VideoCapture(f_source)

    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    write_queue = None
    write_thread = None
    if b_save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(
            "{}/{}.mp4".format(d_save_dir, source_name), fourcc, fps, (w, h)
        )
        write_queue = queue.Queue(maxsize=300)
        write_thread = threading.Thread(
            target=_video_writer_worker, args=(write_queue, video_out), daemon=True
        )
        write_thread.start()

    csv_path = "{}/{}.csv".format(img_save_path, "prediction")
    f_save_txt = open(csv_path, "w")
    f_save_txt.write("frame_num,timestamp_ms,visible,x,y\n")

    count = 0
    pbar = tqdm(total=video_len, desc="Detecting", unit="frame", ncols=100)

    while vid_cap.isOpened():
        imgs = []
        for _ in range(SQ):
            ret, img = vid_cap.read()
            if not ret:
                break
            imgs.append(img)

        if len(imgs) < SQ:
            break

        # numpy でまとめて変換 → 1回の GPU 転送
        frames_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in imgs])
        frames_t = (
            torch.from_numpy(np.ascontiguousarray(frames_np.transpose(0, 3, 1, 2)))
            .to(device, non_blocking=True)
            .to(dtype)
            .div_(255.0)
        )
        frames_t = torchvision.transforms.functional.resize(frames_t, imgsz, antialias=True)
        # 事前確保バッファに copy_() でコピー（GPU malloc を省略）
        input_buffer.copy_(frames_t.view(1, SQ * 3, imgsz[0], imgsz[1]))

        with torch.inference_mode():
            preds = model(input_buffer)  # [1, 3, H, W]

        # GPU 上で閾値処理してから転送
        y_preds = (preds[0] > 0.5).cpu().numpy().astype("uint8") * 255

        for i in range(SQ):
            (visible, cx_pred, cy_pred) = get_shuttle_position(y_preds[i])
            (cx, cy) = (int(cx_pred * w / imgsz[1]), int(cy_pred * h / imgsz[0]))
            timestamp_ms = round(count / fps * 1000, 1)

            if visible:
                cv2.circle(imgs[i], (cx, cy), 8, (0, 0, 255), -1)
                f_save_txt.write("{},{},{},{},{}\n".format(count, timestamp_ms, visible, cx, cy))
            else:
                f_save_txt.write("{},{},{},{},{}\n".format(count, timestamp_ms, visible, "", ""))

            if b_view_img:
                cv2.imwrite("{}/{}.png".format(img_save_path, count), imgs[i])

            if b_save_video:
                write_queue.put(imgs[i])

            count += 1

        pbar.update(SQ)

    pbar.close()

    while count < video_len:
        timestamp_ms = round(count / fps * 1000, 1)
        f_save_txt.write("{},{},0,{},{}\n".format(count, timestamp_ms, "", ""))
        count += 1

    f_save_txt.close()
    print("Saved CSV: {}".format(csv_path))

    if b_save_video:
        write_queue.put(None)
        write_thread.join()
        video_out.release()
        print("Saved video: {}/{}.mp4".format(d_save_dir, source_name))

    vid_cap.release()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
