# XIAO ESP32-S3 Sense — WiFi Camera for SignBridge

Standalone PlatformIO project that turns a [Seeed XIAO ESP32-S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)
into a WiFi-connected MJPEG camera, used by SignBridge's external-camera setting
(Live mode → settings cog → Camera source → WiFi Camera (IP)).

## Wiring / hardware

- **Board**: XIAO ESP32-S3 Sense (ESP32-S3R8, 8 MB OPI PSRAM, 8 MB flash, OV2640 + SD slot + mic on the Sense expansion board)
- **Camera**: OV2640 ribbon plugged into the FPC connector on the Sense board
- **USB**: native USB-CDC — no driver install needed on Windows 10/11

## Setup

1. Open this folder in VS Code with the **PlatformIO IDE** extension installed.
2. Create your WiFi credentials file (gitignored — never committed):
   ```
   cp src/wifi_credentials.example.h src/wifi_credentials.h
   ```
   Then edit `src/wifi_credentials.h` and fill in your hotspot/router `WIFI_SSID` and `WIFI_PASSWORD`.
3. Plug the board in (it should appear as a COM port).
4. PlatformIO sidebar → Project Tasks → **seeed_xiao_esp32s3 → Upload**.
   (Or run `pio run -t upload` from the terminal.)
5. PlatformIO sidebar → Project Tasks → **Monitor** to watch serial output.
   The board prints its IP address and the stream URL on boot.

## Using it with SignBridge

1. Note the printed URL, e.g. `http://192.168.4.123:81/stream`
2. In SignBridge → Live mode → cog → Camera source → **WiFi Camera (IP)**
3. Paste the URL → click **Connect**.

SignBridge proxies the stream through its `/api/camera-proxy` Flask route to
avoid HTTPS-mixed-content issues, so it works even though SignBridge runs over
`wss://` and the camera serves over plain HTTP.

## What's served

- `GET /` — small HTML test page with an embedded `<img src='/stream'>`. Open this in a browser to verify the camera is alive before pointing SignBridge at it.
- `GET /stream` — `multipart/x-mixed-replace; boundary=frame` MJPEG stream. Default frame size 640×480 (VGA), JPEG quality 12. CORS-permissive (`Access-Control-Allow-Origin: *`).

## Tweaking image quality / framerate

Edit `src/main.cpp` `init_camera()`:

| Setting | Effect | Trade-off |
|---|---|---|
| `cfg.frame_size = FRAMESIZE_VGA` (default) | 640×480 | Best balance for SignBridge's 320×240 downsample |
| `FRAMESIZE_SVGA` | 800×600 | Sharper but more bandwidth, lower fps |
| `FRAMESIZE_HVGA` | 480×320 | Lower bandwidth, more fps |
| `cfg.jpeg_quality = 12` | 0 (best) → 63 (worst) | Lower = bigger frames, more bandwidth |

After editing, just re-Upload.

## Phase 2: wide-angle lens

When you swap the default OV2640 for a wide-angle module on the same connector,
no firmware change is required — the pin map is identical. You may want to
re-tune `cfg.brightness` / `cfg.contrast` in `init_camera()` depending on the
new lens's exposure characteristics.
