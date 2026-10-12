#!/usr/bin/env python3
"""Serves index.html and proxies your IP cam at /cam (adds same-origin access so WebGL can read it).
Usage: python3 server.py [camera_url]
Default camera: http://192.168.1.117:8080/video   (IP Webcam app's MJPEG stream)
"""
import os, sys, ssl, urllib.request, threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

CAM = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.117:8080/video"
PORT = int(os.environ.get("PORT", sys.argv[2] if len(sys.argv) > 2 else "8000"))
LOCAL_CAMERA = CAM.isdigit()
LOCAL_CAP = cv2.VideoCapture(int(CAM)) if LOCAL_CAMERA else None
LOCAL_LOCK = threading.Lock()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = DeepLabV3_ResNet101_Weights.DEFAULT
MODEL = deeplabv3_resnet101(weights=WEIGHTS).to(DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def make_deeplab_mask(frame):
    original_height, original_width = frame.shape[:2]
    # Keep enough detail for limbs while fitting comfortably in 4 GB VRAM.
    input_width = 768
    input_height = round(original_height * input_width / original_width)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_width, input_height), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
    image = (image - MEAN) / STD

    with torch.inference_mode():
        logits = MODEL(image)["out"]
        # COCO class 15 is person. Use probability instead of hard argmax
        # so thin limbs survive the mask threshold.
        person_probability = logits.softmax(dim=1)[:, 15:16]
        # Upscale the soft probability first, then threshold. This gives a
        # cleaner boundary than thresholding the low-resolution model output.
        person_probability = F.interpolate(
            person_probability,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        mask = (person_probability[0, 0] > 0.45).mul(255).byte().cpu().numpy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 3)
    return mask

def read_camera_frame():
    if LOCAL_CAMERA:
        with LOCAL_LOCK:
            ok, frame = LOCAL_CAP.read()
        return frame if ok else None
    r = urllib.request.urlopen(CAM, timeout=10, context=ctx)
    data = b""
    while len(data) < 2_000_000:
        chunk = r.read(8192)
        if not chunk:
            break
        data += chunk
        start = data.find(b"\xff\xd8")
        end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
        if start >= 0 and end >= 0:
            return cv2.imdecode(np.frombuffer(data[start:end + 2], np.uint8), cv2.IMREAD_COLOR)
    return None
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/snapshot"):
            try:
                frame = read_camera_frame()
                if frame is None:
                    raise RuntimeError("No camera frame")
                ok, jpeg = cv2.imencode(".jpg", frame)
                payload = jpeg.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(502, str(e))
            return
        if self.path.startswith("/deeplab-mask") or self.path.startswith("/yolo-mask"):
            try:
                frame = read_camera_frame()
                if frame is None:
                    raise RuntimeError("No camera frame")
                mask = make_deeplab_mask(frame)
                mask = cv2.resize(mask, (640, 360), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".png", mask)
                payload = encoded.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(502, str(e))
            return
        if self.path.startswith("/cam"):
            try:
                r = urllib.request.urlopen(CAM, timeout=10, context=ctx)
                self.send_response(200)
                self.send_header("Content-Type", r.headers.get("Content-Type", "multipart/x-mixed-replace"))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    chunk = r.read(8192)
                    if not chunk: break
                    self.wfile.write(chunk)
            except Exception as e:
                try: self.send_error(502, str(e))
                except Exception: pass
        else:
            super().do_GET()
    def log_message(self, *a): pass

print("Camera:", CAM, "\nModel: DeepLabv3-ResNet101 on", DEVICE, "\nOpen http://localhost:" + str(PORT))
ThreadingHTTPServer(("", PORT), H).serve_forever()
