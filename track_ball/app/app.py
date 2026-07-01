import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import tempfile
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request

from models.tracknet import TrackNet
from utils.general import get_shuttle_position

app = Flask(__name__)

_model = None
_device = "cuda"
_imgsz = [288, 512]


def init_model(weights: str, device: str = "cuda", imgsz=None):
    global _model, _device, _imgsz
    if imgsz:
        _imgsz = imgsz
    _device = device
    model = TrackNet().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    _model = model
    print(f"モデルロード完了: {weights}")


def _run_prediction(video, model, device, imgsz, out_path="./predict.mp4"):
    vid_cap = cv2.VideoCapture(video)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    count = 0
    video_end = False
    while vid_cap.isOpened():
        imgs = []
        for _ in range(3):
            ret, img = vid_cap.read()
            if not ret:
                video_end = True
                break
            imgs.append(img)

        if video_end:
            break

        imgs_torch = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_torch = transforms.ToTensor()(img).to(device)
            img_torch = transforms.functional.resize(img_torch, imgsz, antialias=True)
            imgs_torch.append(img_torch)

        imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

        preds = model(imgs_torch)
        preds = preds[0].detach().cpu().numpy()

        y_preds = (preds > 0.5).astype('float32') * 255
        y_preds = y_preds.astype('uint8')

        for i in range(3):
            (visible, cx_pred, cy_pred) = get_shuttle_position(y_preds[i])
            cx = int(cx_pred * w / imgsz[1])
            cy = int(cy_pred * h / imgsz[0])
            if visible:
                cv2.circle(imgs[i], (cx, cy), 8, (0, 0, 255), -1)
            out.write(imgs[i])
            print(f"{count} ---- visible: {visible}  cx: {cx}  cy: {cy}")
            count += 1

    out.release()
    vid_cap.release()


@app.route('/predict', methods=['POST'])
def predict():
    if _model is None:
        return jsonify({'error': 'モデル未初期化。init_model() を先に呼んでください。'}), 503

    file = request.files['file']

    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp:
        temp.write(file.read())
        temp.flush()  # ディスクへ書き出してから OpenCV に渡す
        out_path = temp.name.replace('.mp4', '_out.mp4')
        _run_prediction(temp.name, _model, _device, _imgsz, out_path)

    return 'Video processed successfully'


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512])
    parser.add_argument('--weights', type=str, default=str(ROOT / 'best.pt'))
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    init_model(str(opt.weights), imgsz=opt.imgsz)
    app.run()
