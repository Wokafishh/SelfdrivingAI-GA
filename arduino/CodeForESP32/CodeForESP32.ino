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
const int    TARGET_PORT = 8000;
const int    CONTROL_PORT = 8001;

const int LED_PIN_1 = A0;
const int LED_PIN_2 = A1;

const size_t MAX_PAYLOAD_SIZE = 50000;
const size_t UDP_CHUNK_SIZE   = 1200;

uint8_t payloadBuffer[MAX_PAYLOAD_SIZE];
uint8_t chunkTxBuffer[UDP_CHUNK_SIZE + 4];

IPAddress cachedServerIP(0, 0, 0, 0);

// --- Dual Streaming Mode & Motor Control State ---
bool isContinuousMode = true;
bool frameRequested    = true;
float currentVal1      = 0.0f; // Left / Axis A (-1.0 to 1.0)
float currentVal2      = 0.0f; // Right / Axis B (-1.0 to 1.0)

// --- Diagnostics & Performance Metrics ---
uint32_t totalFramesSent = 0;
uint32_t totalBytesSent  = 0;
uint32_t camErrorCount   = 0;
uint32_t udpErrorCount   = 0;

uint32_t framesInWindow    = 0;
uint32_t bytesInWindow     = 0;
uint32_t lastStatsTime     = 0;
uint32_t lastMdnsRetryTime = 0;
bool     isWifiConnected   = false;

// Forward declaration
void updateLampIndicators();
void pollControlPacket();

void pulseLED(int pin, uint32_t durationMs) {
  digitalWrite(pin, HIGH);
  delay(durationMs);
  digitalWrite(pin, LOW);
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
  uint32_t tStart = micros();

  while (Serial1.available()) {
    Serial1.read();
  }

  Serial1.println("CAPTURE");

  String header = Serial1.readStringUntil('\n');
  header.trim();

  if (!header.startsWith("SIZE:")) {
    camErrorCount++;
    return -1;
  }

  int imgSize = header.substring(5).toInt();
  if (imgSize <= 0 || (size_t)imgSize > maxLen) {
    camErrorCount++;
    return -1;
  }

  size_t bytesRead = Serial1.readBytes((char*)buffer, imgSize);
  if (bytesRead != (size_t)imgSize) {
    camErrorCount++;
    return -1;
  }

  return imgSize;
}

void pollControlPacket() {
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    char packetBuffer[64];
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    packetBuffer[len] = '\0';

    float v1 = 0.0f;
    float v2 = 0.0f;
    char modeChar = 'C';

    if (sscanf(packetBuffer, "%f,%f,%c", &v1, &v2, &modeChar) >= 2) {
      currentVal1 = v1;
      currentVal2 = v2;

      if (modeChar == 'I') {
        isContinuousMode = false;
      } else if (modeChar == 'C') {
        isContinuousMode = true;
      }
      frameRequested = true;
      updateLampIndicators(); // Instant reaction
    }
  }
}

void updateLampIndicators() {
  bool isIdle = (currentVal1 == 0.0f && currentVal2 == 0.0f);
  bool isReversing = (currentVal1 < 0.0f && currentVal2 < 0.0f);

  if (isIdle) {
    digitalWrite(LED_PIN_1, LOW);
    digitalWrite(LED_PIN_2, LOW);
  } else if (isReversing) {
    static uint32_t lastRevBlink = 0;
    static bool revBlinkState = false;
    if (millis() - lastRevBlink >= 250) {
      lastRevBlink = millis();
      revBlinkState = !revBlinkState;
    }
    digitalWrite(LED_PIN_1, revBlinkState ? HIGH : LOW);
    digitalWrite(LED_PIN_2, revBlinkState ? HIGH : LOW);
  } else {
    digitalWrite(LED_PIN_1, currentVal1 > 0 ? HIGH : LOW);
    digitalWrite(LED_PIN_2, currentVal2 > 0 ? HIGH : LOW);
  }
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
    // Poll for control packets between chunks for zero-lag input response
    pollControlPacket();
    updateLampIndicators();

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

    Serial.println("\n------------- [LIVE SYSTEM METRICS] -------------");
    Serial.printf("  FPS:               %.1f frames/sec\n", fps);
    Serial.printf("  Streaming Mode:    %s\n", isContinuousMode ? "CONTINUOUS" : "LOCKSTEP (ON-INPUT)");
    Serial.printf("  Motor Cmd (Tanh):  %.1f, %.1f\n", currentVal1, currentVal2);
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
  delay(1000);
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  Serial1.setRxBufferSize(16384);
  Serial1.setTimeout(200);
  Serial1.begin(1000000, SERIAL_8N1, D0, D1);

  pinMode(LED_PIN_1, OUTPUT);
  pinMode(LED_PIN_2, OUTPUT);
  digitalWrite(LED_PIN_1, LOW);
  digitalWrite(LED_PIN_2, LOW);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  for (size_t i = 0; i < numNetworks; i++) {
    wifiMulti.addAP(wifiNetworks[i].ssid, wifiNetworks[i].password);
  }

  uint32_t lastBlinkTime = 0;
  bool blinkState = false;

  while (wifiMulti.run() != WL_CONNECTED) {
    if (millis() - lastBlinkTime >= 500) {
      lastBlinkTime = millis();
      blinkState = !blinkState;
      digitalWrite(LED_PIN_1, blinkState ? HIGH : LOW);
    }
    delay(20);
  }

  digitalWrite(LED_PIN_1, LOW);
  pulseLED(LED_PIN_2, 200);

  isWifiConnected = true;

  MDNS.begin("nano-esp32");
  udp.begin(CONTROL_PORT);

  lastStatsTime = millis();
}

void loop() {
  if (wifiMulti.run() != WL_CONNECTED) {
    if (isWifiConnected) {
      isWifiConnected = false;
      cachedServerIP = IPAddress(0, 0, 0, 0);
    }
    delay(20);
    return;
  } else if (!isWifiConnected) {
    isWifiConnected = true;
    digitalWrite(LED_PIN_1, LOW);
    pulseLED(LED_PIN_2, 200);
    frameRequested = true;
  }

  IPAddress serverIP = resolveServerIP();
  if (serverIP == IPAddress(0, 0, 0, 0)) {
    delay(50);
    return;
  }

  // 3. Poll control packets immediately at the start of loop
  pollControlPacket();
  updateLampIndicators();

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

  updateLampIndicators();
  yield();
}
