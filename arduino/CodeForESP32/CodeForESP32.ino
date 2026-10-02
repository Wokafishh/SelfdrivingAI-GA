#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>

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

const int LED_PIN_1 = A0; // Blinks slowly during connection
const int LED_PIN_2 = A1; // Pulses once on successful connection

const size_t MAX_PAYLOAD_SIZE = 50000;
const size_t UDP_CHUNK_SIZE   = 1200; // Keep below 1400 MTU limit

uint8_t payloadBuffer[MAX_PAYLOAD_SIZE];
IPAddress cachedServerIP(0, 0, 0, 0);

// --- Dual Streaming Mode Control ---
bool isContinuousMode = true; // Default: Continuous streaming when toggle is OFF
bool frameRequested   = true; // Single-shot trigger when toggle is ON

// --- Diagnostics & Performance Metrics ---
uint32_t totalFramesSent = 0;
uint32_t totalBytesSent  = 0;
uint32_t camErrorCount   = 0;
uint32_t udpErrorCount   = 0;

// Windowed metrics for FPS / Throughput calculation
uint32_t framesInWindow    = 0;
uint32_t bytesInWindow     = 0;
uint32_t lastStatsTime     = 0;
uint32_t lastMdnsRetryTime = 0;
bool     isWifiConnected   = false;

// LED Blink Helper Function
void pulseLED(int pin, uint32_t durationMs) {
  digitalWrite(pin, HIGH);
  delay(durationMs);
  digitalWrite(pin, LOW);
}

IPAddress resolveServerIP() {
  if (cachedServerIP != IPAddress(0, 0, 0, 0)) return cachedServerIP;

  if (millis() - lastMdnsRetryTime < 2000) return cachedServerIP;
  lastMdnsRetryTime = millis();

  Serial.printf("[mDNS] Resolving target host '%s'...\n", TARGET_HOST);

  IPAddress ip = MDNS.queryHost(TARGET_HOST);
  if (ip == IPAddress(0, 0, 0, 0)) {
    ip = MDNS.queryHost("larvtop.local");
  }

  if (ip != IPAddress(0, 0, 0, 0)) {
    cachedServerIP = ip;
    Serial.printf("[mDNS SUCCESS] Resolved IP: %s\n", cachedServerIP.toString().c_str());
  } else {
    Serial.println("[mDNS WARN] Host resolution failed. Retrying...");
  }
  return cachedServerIP;
}

int fetchFrameFromCamera(uint8_t* buffer, size_t maxLen) {
  uint32_t tStart = micros();

  // Clear leftover garbage bytes in RX buffer
  size_t flushedBytes = 0;
  while (Serial1.available()) {
    Serial1.read();
    flushedBytes++;
  }

  Serial1.println("CAPTURE");

  String header = Serial1.readStringUntil('\n');
  header.trim();

  if (!header.startsWith("SIZE:")) {
    camErrorCount++;
    Serial.printf("[CAM ERR] Corrupted header: '%s' (Flushed %u rx bytes)\n", header.c_str(), (unsigned int)flushedBytes);
    return -1;
  }

  int imgSize = header.substring(5).toInt();
  if (imgSize <= 0 || (size_t)imgSize > maxLen) {
    camErrorCount++;
    Serial.printf("[CAM ERR] Invalid frame size: %d bytes (Max limit: %u)\n", imgSize, (unsigned int)maxLen);
    return -1;
  }

  size_t bytesRead = Serial1.readBytes((char*)buffer, imgSize);
  uint32_t tElapsedUs = micros() - tStart;

  if (bytesRead != (size_t)imgSize) {
    camErrorCount++;
    Serial.printf("[CAM ERR] UART Timeout! Received %u of %d bytes after %.2f ms\n",
                  (unsigned int)bytesRead, imgSize, tElapsedUs / 1000.0);
    return -1;
  }

  return imgSize;
}

void setup() {
  delay(1000);
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  Serial.println("\n===========================================");
  Serial.println("  ESP32 High-Speed UDP Streamer + Debug    ");
  Serial.println("===========================================");

  Serial1.setRxBufferSize(16384);
  Serial1.setTimeout(200);
  Serial1.begin(1000000, SERIAL_8N1, D0, D1);
  Serial.println("[HW] Camera UART initialized @ 1,000,000 baud.");

  pinMode(LED_PIN_1, OUTPUT);
  pinMode(LED_PIN_2, OUTPUT);
  digitalWrite(LED_PIN_1, LOW);
  digitalWrite(LED_PIN_2, LOW);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  Serial.println("[WiFi] Disabled WiFi modem power-saving sleep.");

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
      digitalWrite(LED_PIN_1, blinkState ? HIGH : LOW);
      Serial.print(".");
    }
    delay(20);
  }

  digitalWrite(LED_PIN_1, LOW);
  pulseLED(LED_PIN_2, 200);

  isWifiConnected = true;
  Serial.printf("\n[WiFi CONNECTED] SSID: %s | Local IP: %s | RSSI: %d dBm\n",
                WiFi.SSID().c_str(), WiFi.localIP().toString().c_str(), WiFi.RSSI());

  if (MDNS.begin("nano-esp32")) {
    Serial.println("[mDNS] Local responder ready: nano-esp32.local");
  } else {
    Serial.println("[mDNS ERR] Responder initialization failed!");
  }

  udp.begin(CONTROL_PORT);
  Serial.printf("[UDP] Control socket listening on port %d\n", CONTROL_PORT);

  lastStatsTime = millis();
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
      continue;
    }

    uint8_t chunkHeader[4] = { frameId, chunkIdx, totalChunks, counter };
    udp.write(chunkHeader, 4);
    udp.write(payloadBuffer + offset, chunkSize);

    if (!udp.endPacket()) {
      udpErrorCount++;
      frameHadError = true;
      // Socket Recovery: Re-bind socket if transmission fails to prevent permanent lockup
      udp.stop();
      udp.begin(CONTROL_PORT);
      break;
    }
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

    Serial.println("\n------------- [LIVE SYSTEM METRICS] -------------");
    Serial.printf("  FPS:              %.1f frames/sec\n", fps);
    Serial.printf("  Streaming Mode:  %s\n", isContinuousMode ? "CONTINUOUS" : "LOCKSTEP (ON-INPUT)");
    Serial.printf("  Data Throughput: %.2f KB/s (%.2f Mbps)\n", kbps, (kbps * 8.0) / 1024.0);
    Serial.printf("  Avg Frame Size:  %.1f KB\n", avgFrameKB);
    Serial.printf("  Wi-Fi Status:    %s | RSSI: %d dBm | SSID: %s\n",
                  (WiFi.status() == WL_CONNECTED) ? "ONLINE" : "OFFLINE",
                  WiFi.RSSI(), WiFi.SSID().c_str());
    Serial.printf("  Target Server:   %s:%d\n", cachedServerIP.toString().c_str(), TARGET_PORT);
    Serial.printf("  Total Stats:     Sent: %u frames (%u KB) | Cam Errors: %u | UDP Errors: %u\n",
                  totalFramesSent, totalBytesSent / 1024, camErrorCount, udpErrorCount);
    Serial.println("-------------------------------------------------");

    framesInWindow = 0;
    bytesInWindow  = 0;
    lastStatsTime  = now;
  }
}

void loop() {
  // 1. Wi-Fi Connection Guard
  if (wifiMulti.run() != WL_CONNECTED) {
    if (isWifiConnected) {
      isWifiConnected = false;
      cachedServerIP = IPAddress(0, 0, 0, 0);
      Serial.println("\n[WiFi WARN] Connection dropped! Attempting auto-reconnect...");
    }

    static uint32_t lastLoopBlink = 0;
    static bool loopBlinkState = false;
    if (millis() - lastLoopBlink >= 500) {
      lastLoopBlink = millis();
      loopBlinkState = !loopBlinkState;
      digitalWrite(LED_PIN_1, loopBlinkState ? HIGH : LOW);
    }

    delay(20);
    return;
  } else if (!isWifiConnected) {
    isWifiConnected = true;
    digitalWrite(LED_PIN_1, LOW);
    pulseLED(LED_PIN_2, 200);

    Serial.printf("\n[WiFi RECONNECTED] SSID: %s | IP: %s | RSSI: %d dBm\n",
                  WiFi.SSID().c_str(), WiFi.localIP().toString().c_str(), WiFi.RSSI());

    frameRequested = true;
  }

  // 2. Ensure target server IP is resolved
  IPAddress serverIP = resolveServerIP();
  if (serverIP == IPAddress(0, 0, 0, 0)) {
    delay(50);
    return;
  }

  // 3. Listen for incoming control commands
  int packetSize = udp.parsePacket();
  if (packetSize >= 2) {
    char led1 = udp.read();
    char led2 = udp.read();
    digitalWrite(LED_PIN_1, led1 == '1' ? HIGH : LOW);
    digitalWrite(LED_PIN_2, led2 == '1' ? HIGH : LOW);

    // Read mode byte: 'C' = Continuous, 'I' = Lockstep On-Input
    if (packetSize >= 3) {
      char modeChar = udp.read();
      if (modeChar == 'I') {
        isContinuousMode = false;
      } else if (modeChar == 'C') {
        isContinuousMode = true;
      }
    }

    // Authorize next single-shot capture
    frameRequested = true;
  }

  // 4. Send frame if continuously streaming OR if authorized by an input
  if (isContinuousMode || frameRequested) {
    int imgBytes = fetchFrameFromCamera(payloadBuffer, MAX_PAYLOAD_SIZE);
    if (imgBytes > 0) {
      sendFrameUDP(serverIP, imgBytes);
    }
    frameRequested = false;
  }

  // 5. Output periodic health metrics
  printRecurringStats();

  yield();
}
