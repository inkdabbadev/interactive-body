# Interactive Body Particles

DeepLabv3 human silhouette extraction on CUDA, rendered as WebGL particles,
glossy balls, or balloons.

## Run

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Start with an IP Webcam MJPEG stream:

```bash
PORT=8000 python server.py http://PHONE_IP:8080/video
```

Open http://localhost:8000.

The DeepLabv3 model weights download automatically on first start. CUDA is
used automatically when available; CPU fallback is supported.

## Controls

- `W`: white background with colorful balls
- `N`: black background with blue particles
- `P`: particles
- `B`: balls
- `V`: balloons
- Hold left or right click for 0.5 seconds to create a crack effect
- Move the pointer for cherry-blossom trails
