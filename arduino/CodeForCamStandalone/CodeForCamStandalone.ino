#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// --- ESP32-CAM Pin Definitions & LED Helpers ---
#define LED_BUILTIN_CAM 33 // Onboard red LED (GPIO 33, LOW = ON)
#define LED_FLASH 4        // Onboard flash LED (GPIO 4, HIGH = ON)

const int LED_PIN_1 = LED_BUILTIN_CAM;
const int LED_PIN_2 = LED_FLASH;

// --- Wi-Fi & Network Configuration ---
WiFiMulti wifiMulti;
WiFiUDP udp;

struct WiFiCredential {
  const char* ssid;
  const char* password;
};

const WiFiCredential wifiNetworks[] = {
  {"heddaMakes", "h3dd4m4k3s"},
  {"TP-Link_0958", "95704883"},
  {"LarvFi", "IngetLosenord"},
};
const size_t numNetworks = sizeof(wifiNetworks) / sizeof(wifiNetworks[0]);

const char* TARGET_HOST = "larvtop";
const int   TARGET_PORT = 8000;
const int   CONTROL_PORT = 8001;

const size_t MAX_PAYLOAD_SIZE = 50000;
const size_t UDP_CHUNK_SIZE   = 1200; // Prevents packet fragmentation

uint8_t payloadBuffer[MAX_PAYLOAD_SIZE];
uint8_t chunkTxBuffer[UDP_CHUNK_SIZE + 4];

IPAddress cachedServerIP(0, 0, 0, 0);

// --- Dual Streaming Mode Control ---
bool isContinuousMode = true;
bool frameRequested    = true;

// --- Diagnostics & Metrics ---
uint32_t totalFramesSent = 0;
uint32_t totalBytesSent  = 0;
uint32_t camErrorCount   = 0;
uint32_t udpErrorCount   = 0;

uint32_t framesInWindow    = 0;
uint32_t bytesInWindow     = 0;
uint32_t lastStatsTime     = 0;
uint32_t lastMdnsRetryTime = 0;
bool     isWifiConnected   = false;

unsigned long lastCaptureTime = 0;

// --- Camera & LED Helpers ---
void warmupCamera(int frames) {
  for (int i = 0; i < frames; i++) {
    camera_fb_t * dummy = esp_camera_fb_get();
    if (dummy) esp_camera_fb_return(dummy);
    delay(30);
  }
}

void setLed1(bool state) {
  digitalWrite(LED_PIN_1, state ? LOW : HIGH); // Active LOW on ESP32-CAM
}

void setLed2(bool state) {
  digitalWrite(LED_PIN_2, state ? HIGH : LOW); // Active HIGH
}

void pulseLED(int pin, uint32_t durationMs) {
  if (pin == LED_BUILTIN_CAM) {
    digitalWrite(pin, LOW);
    delay(durationMs);
    digitalWrite(pin, HIGH);
  } else {
    digitalWrite(pin, HIGH);
    delay(durationMs);
    digitalWrite(pin, LOW);
  }
}

IPAddress resolveServerIP() {
  if (cachedServerIP != IPAddress(0, 0, 0, 0)) return cachedServerIP;
  if (millis() - lastMdnsRetryTime < 2000) return cachedServerIP;
  lastMdnsRetryTime = millis();

  IPAddress ip = MDNS.queryHost(TARGET_HOST);
  if (ip == IPAddress(0, 0, 0, 0)) {
    ip = MDNS.queryHost("larvtop.local");
  }

  if (ip != IPAddress(0, 0, 0, 0)) {
    cachedServerIP = ip;
  }
  return cachedServerIP;
}

int fetchFrameFromCamera(uint8_t* buffer, size_t maxLen) {
  if (millis() - lastCaptureTime > 3000) {
    warmupCamera(2);
  }

  setLed1(true); // Turn on red LED during capture

  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    camErrorCount++;
    setLed1(false);
    return -1;
  }

  lastCaptureTime = millis();

  if (fb->len > maxLen) {
    camErrorCount++;
    esp_camera_fb_return(fb);
    setLed1(false);
    return -1;
  }

  memcpy(buffer, fb->buf, fb->len);
  size_t imgSize = fb->len;

  esp_camera_fb_return(fb);
  setLed1(false); // Turn off red LED

  return imgSize;
}

void sendFrameUDP(IPAddress serverIP, int imgBytes) {
  static uint8_t frameId = 0;
  static uint8_t counter = 0;
  frameId++;
  counter++;

  uint16_t totalLen = imgBytes;
  uint8_t totalChunks = (totalLen + UDP_CHUNK_SIZE - 1) / UDP_CHUNK_SIZE;
  bool frameHadError = false;

  for (uint8_t chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    size_t offset = chunkIdx * UDP_CHUNK_SIZE;
    size_t bytesLeft = totalLen - offset;
    size_t chunkSize = (bytesLeft < UDP_CHUNK_SIZE) ? bytesLeft : UDP_CHUNK_SIZE;

    if (!udp.beginPacket(serverIP, TARGET_PORT)) {
      udpErrorCount++;
      frameHadError = true;
      break;
    }

    chunkTxBuffer[0] = frameId;
    chunkTxBuffer[1] = chunkIdx;
    chunkTxBuffer[2] = totalChunks;
    chunkTxBuffer[3] = counter;

    memcpy(chunkTxBuffer + 4, payloadBuffer + offset, chunkSize);
    udp.write(chunkTxBuffer, chunkSize + 4);

    if (!udp.endPacket()) {
      udpErrorCount++;
      frameHadError = true;
      break;
    }

    delayMicroseconds(150);
  }

  if (!frameHadError) {
    totalFramesSent++;
    totalBytesSent += imgBytes;
    framesInWindow++;
    bytesInWindow += imgBytes;
  }
}

void printRecurringStats() {
  uint32_t now = millis();
  uint32_t elapsedMs = now - lastStatsTime;

  if (elapsedMs >= 2000) {
    float seconds    = elapsedMs / 1000.0;
    float fps        = framesInWindow / seconds;
    float kbps       = (bytesInWindow / 1024.0) / seconds;
    float avgFrameKB = framesInWindow > 0 ? (bytesInWindow / 1024.0) / framesInWindow : 0.0;

    Serial.println("\n------------- [ESP32-CAM LIVE METRICS] -------------");
    Serial.printf("  FPS:               %.1f frames/sec\n", fps);
    Serial.printf("  Streaming Mode:    %s\n", isContinuousMode ? "CONTINUOUS" : "LOCKSTEP (ON-INPUT)");
    Serial.printf("  Data Throughput:   %.2f KB/s (%.2f Mbps)\n", kbps, (kbps * 8.0) / 1024.0);
    Serial.printf("  Avg Frame Size:    %.1f KB\n", avgFrameKB);
    Serial.printf("  Wi-Fi Status:      %s | RSSI: %d dBm | SSID: %s\n",
                  (WiFi.status() == WL_CONNECTED) ? "ONLINE" : "OFFLINE",
                  WiFi.RSSI(), WiFi.SSID().c_str());
    Serial.printf("  Target Server:     %s:%d\n", cachedServerIP.toString().c_str(), TARGET_PORT);
    Serial.printf("  Total Stats:       Sent: %u frames (%u KB) | Cam Errors: %u | UDP Errors: %u\n",
                  totalFramesSent, totalBytesSent / 1024, camErrorCount, udpErrorCount);
    Serial.println("-------------------------------------------------");

    framesInWindow = 0;
    bytesInWindow  = 0;
    lastStatsTime  = now;
  }
}

void setup() {
  // Disable the overly aggressive software brownout detector
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  delay(1000);
  Serial.begin(115200);

  // Initialize external PSRAM required for ESP32-CAM frame buffers
  if (psramInit()) {
    Serial.println("[HW] PSRAM initialized successfully.");
  } else {
    Serial.println("[HW WARN] PSRAM initialization failed or not present!");
  }

  Serial.println("\n===========================================");
  Serial.println("     ESP32-CAM Direct WiFi UDP Streamer    ");
  Serial.println("===========================================");

  pinMode(LED_PIN_1, OUTPUT);
  pinMode(LED_PIN_2, OUTPUT);
  setLed1(false);
  setLed2(false);

  // --- Initialize Camera ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;    config.pin_d1 = 18;  config.pin_d2 = 19;
  config.pin_d3 = 21;   config.pin_d4 = 36;  config.pin_d5 = 39;
  config.pin_d6 = 34;   config.pin_d7 = 35;  config.pin_xclk = 0;
  config.pin_pclk = 22; config.pin_vsync = 25; config.pin_href = 23;
  config.pin_sscb_sda = 26; config.pin_sscb_scl = 27; config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_96X96;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("ERR:CAM_INIT_FAILED_0x%x\n", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 2);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
  }

  warmupCamera(5);
  Serial.println("[HW] Camera initialized successfully.");

  // --- Initialize Wi-Fi ---
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  for (size_t i = 0; i < numNetworks; i++) {
    wifiMulti.addAP(wifiNetworks[i].ssid, wifiNetworks[i].password);
  }

  Serial.print("[WiFi] Connecting");
  uint32_t lastBlinkTime = 0;
  bool blinkState = false;

  while (wifiMulti.run() != WL_CONNECTED) {
    if (millis() - lastBlinkTime >= 500) {
      lastBlinkTime = millis();
      blinkState = !blinkState;
      setLed1(blinkState);
      Serial.print(".");
    }
    delay(20);
  }

  setLed1(false);
  pulseLED(LED_PIN_2, 200);

  isWifiConnected = true;
  Serial.printf("\n[WiFi CONNECTED] SSID: %s | Local IP: %s | RSSI: %d dBm\n",
                WiFi.SSID().c_str(), WiFi.localIP().toString().c_str(), WiFi.RSSI());

  if (MDNS.begin("esp32-cam")) {
    Serial.println("[mDNS] Local responder ready: esp32-cam.local");
  }

  udp.begin(CONTROL_PORT);
  Serial.printf("[UDP] Control socket listening on port %d\n", CONTROL_PORT);

  // Ready blink
  for (int i = 0; i < 3; i++) {
    setLed1(true);
    delay(100);
    setLed1(false);
    delay(100);
  }

  lastStatsTime = millis();
}

void loop() {
  if (wifiMulti.run() != WL_CONNECTED) {
    if (isWifiConnected) {
      isWifiConnected = false;
      cachedServerIP = IPAddress(0, 0, 0, 0);
    }
    static uint32_t lastLoopBlink = 0;
    static bool loopBlinkState = false;
    if (millis() - lastLoopBlink >= 500) {
      lastLoopBlink = millis();
      loopBlinkState = !loopBlinkState;
      setLed1(loopBlinkState);
    }
    delay(20);
    return;
  } else if (!isWifiConnected) {
    isWifiConnected = true;
    setLed1(false);
    pulseLED(LED_PIN_2, 200);
    frameRequested = true;
  }

  IPAddress serverIP = resolveServerIP();
  if (serverIP == IPAddress(0, 0, 0, 0)) {
    delay(50);
    return;
  }

  int packetSize = udp.parsePacket();
  if (packetSize >= 2) {
    char led1 = udp.read();
    char led2 = udp.read();
    setLed1(led1 == '1');
    setLed2(led2 == '1');

    if (packetSize >= 3) {
      char modeChar = udp.read();
      if (modeChar == 'I') {
        isContinuousMode = false;
      } else if (modeChar == 'C') {
        isContinuousMode = true;
      }
    }
    frameRequested = true;
  }

  if (isContinuousMode || frameRequested) {
    int imgBytes = fetchFrameFromCamera(payloadBuffer, MAX_PAYLOAD_SIZE);
    if (imgBytes > 0) {
      sendFrameUDP(serverIP, imgBytes);
    }
    frameRequested = false;
  }

  printRecurringStats();
  yield();
}
