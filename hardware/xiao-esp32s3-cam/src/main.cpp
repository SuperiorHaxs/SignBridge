// XIAO ESP32-S3 Sense — WiFi MJPEG camera for SignBridge.
//
// Boots, joins WiFi, initializes the OV2640 camera, and serves MJPEG on
//     http://<board-ip>:81/stream
// Plus a tiny test page at
//     http://<board-ip>:81/
// so you can visually verify the stream from a browser before pointing
// SignBridge's external-camera setting at it.

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>

// ── WiFi credentials ──────────────────────────────────────────────────────
// Real values live in src/wifi_credentials.h (gitignored). On a fresh
// checkout: cp src/wifi_credentials.example.h src/wifi_credentials.h
// then edit the new file with your hotspot SSID + password.
#include "wifi_credentials.h"

// ── XIAO ESP32-S3 Sense camera pin map (Seeed reference design, OV2640) ──
// Reference: https://wiki.seeedstudio.com/xiao_esp32s3_camera_usage/
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ── MJPEG framing ─────────────────────────────────────────────────────────
#define PART_BOUNDARY "frame"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART_FMT     = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static httpd_handle_t stream_httpd = NULL;

// /stream — the MJPEG endpoint SignBridge will load.
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = nullptr;
    esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    // CORS: the SignBridge canvas-bridge fetch needs anonymous cross-origin.
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "15");

    char hdr_buf[80];
    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[cam] frame grab failed");
            res = ESP_FAIL;
            break;
        }
        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }

        int hlen = snprintf(hdr_buf, sizeof(hdr_buf), STREAM_PART_FMT, (unsigned)fb->len);
        res = httpd_resp_send_chunk(req, hdr_buf, hlen);
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }

        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;
        // No artificial throttle -- the camera + JPEG encode pipeline naturally
        // caps at ~15-25 fps at VGA on the OV2640.
    }
    return res;
}

// Tiny landing page so you can sanity-check the stream from a browser.
static esp_err_t root_handler(httpd_req_t *req) {
    const char html[] =
        "<!doctype html><html><head><title>XIAO ESP32-S3 Cam</title>"
        "<style>body{background:#0f1419;color:#e6e6e6;font-family:system-ui;text-align:center;margin:0;padding:20px}"
        "img{max-width:640px;width:100%;border-radius:6px;border:1px solid #333}</style></head>"
        "<body><h2>XIAO ESP32-S3 Sense Camera</h2>"
        "<p>Stream URL: <code id='u'></code></p>"
        "<img src='/stream'>"
        "<script>document.getElementById('u').textContent=location.origin+'/stream'</script>"
        "</body></html>";
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html, sizeof(html) - 1);
}

static void start_http_server() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 81;
    config.ctrl_port   = 32769;   // distinct from the default 32768
    config.stack_size  = 8192;
    config.max_uri_handlers = 4;

    httpd_uri_t root_uri = {
        .uri = "/", .method = HTTP_GET, .handler = root_handler, .user_ctx = NULL
    };
    httpd_uri_t stream_uri = {
        .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL
    };

    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &root_uri);
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        Serial.printf("[http] server up on port %d\n", config.server_port);
    } else {
        Serial.println("[http] server FAILED to start");
    }
}

static bool init_camera() {
    camera_config_t cfg = {};
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0       = Y2_GPIO_NUM;
    cfg.pin_d1       = Y3_GPIO_NUM;
    cfg.pin_d2       = Y4_GPIO_NUM;
    cfg.pin_d3       = Y5_GPIO_NUM;
    cfg.pin_d4       = Y6_GPIO_NUM;
    cfg.pin_d5       = Y7_GPIO_NUM;
    cfg.pin_d6       = Y8_GPIO_NUM;
    cfg.pin_d7       = Y9_GPIO_NUM;
    cfg.pin_xclk     = XCLK_GPIO_NUM;
    cfg.pin_pclk     = PCLK_GPIO_NUM;
    cfg.pin_vsync    = VSYNC_GPIO_NUM;
    cfg.pin_href     = HREF_GPIO_NUM;
    cfg.pin_sccb_sda = SIOD_GPIO_NUM;
    cfg.pin_sccb_scl = SIOC_GPIO_NUM;
    cfg.pin_pwdn     = PWDN_GPIO_NUM;
    cfg.pin_reset    = RESET_GPIO_NUM;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    // VGA (640x480) at quality=15 -- 4x more pixels than QVGA so MediaPipe
    // Hands can actually find a hand at lanyard / conversational distance
    // (~3 ft signer). SignBridge client also captures at 480x360 to match,
    // otherwise the resolution gain is thrown away in client-side downscale.
    // Bandwidth ~250 KB/s @ 15fps; needs the external IPEX antenna and a
    // stable hotspot to sustain (PCB antenna + body proximity will stall).
    // Drop back to FRAMESIZE_QVGA + q=12 if WiFi reliability becomes the
    // limiting factor again.
    cfg.frame_size   = FRAMESIZE_VGA;
    cfg.jpeg_quality = 15;
    // Single buffer with grab_latest: never queue, never lag. Better than
    // double-buffering when the network can't drain frames as fast as the
    // camera produces them.
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        Serial.printf("[cam] esp_camera_init failed: 0x%x\n", err);
        return false;
    }
    // Optional tweaks once the sensor is up.
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);
        s->set_saturation(s, 0);
        // Case mounts the board upside down. vflip + hmirror together rotate
        // the OV2640 output 180 deg in hardware (no CPU cost). Flip both
        // back to 0 if the case orientation ever changes.
        s->set_hmirror(s, 1);
        s->set_vflip(s, 1);
    }
    Serial.println("[cam] init OK");
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(500);   // give the USB-CDC interface a moment to attach
    Serial.println();
    Serial.println("=== XIAO ESP32-S3 Sense WiFi Camera ===");

    if (!init_camera()) {
        Serial.println("Halting -- camera unavailable.");
        while (true) delay(1000);
    }

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);   // keep latency low; small power cost is fine when USB-powered
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("Connecting to WiFi '%s'", WIFI_SSID);
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t0 > 30000) {
            Serial.println("\n[wifi] timed out after 30s -- check SSID/password");
            t0 = millis();
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        }
        delay(400);
        Serial.print(".");
    }
    Serial.println();
    IPAddress ip = WiFi.localIP();
    Serial.printf("[wifi] connected, IP=%s, RSSI=%d dBm\n",
                  ip.toString().c_str(), WiFi.RSSI());
    Serial.printf("Stream URL:    http://%s:81/stream\n", ip.toString().c_str());
    Serial.printf("Test page:     http://%s:81/\n", ip.toString().c_str());

    start_http_server();
}

void loop() {
    // All work happens in the HTTP server task. Poll WiFi here for reconnect.
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[wifi] dropped, reconnecting...");
        WiFi.disconnect();
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        delay(2000);
    } else {
        delay(2000);
    }
}
