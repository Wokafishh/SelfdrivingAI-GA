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

// --- Motor Control Pins ---
const int RIGHT_MOTOR_FWD  = A0;
const int RIGHT_MOTOR_BACK = A1;
const int LEFT_MOTOR_FWD   = A2;
const int LEFT_MOTOR_BACK  = A3;

const size_t MAX_PAYLOAD_SIZE = 50000;
const size_t UDP_CHUNK_SIZE   = 1400;
const size_t UDP_HEADER_SIZE  = 12; // [0]:frameId, [1]:chunkIdx, [2]:totalChunks, [3]:counter, [4-7]:captureUs, [8-11]:txUs

uint8_t payloadBuffer[MAX_PAYLOAD_SIZE];
uint8_t chunkTxBuffer[UDP_CHUNK_SIZE + UDP_HEADER_SIZE];

IPAddress cachedServerIP(0, 0, 0, 0);

// --- Dual Streaming Mode & Motor Control State ---
bool isContinuousMode = true;
bool frameRequested    = true;
float currentVal1      = 0.0f;
float currentVal2      = 0.0f;
int   lastSentLightState = -1;

// --- Diagnostics & Performance Metrics ---
uint32_t totalFramesAttempted = 0;
uint32_t totalFramesSent = 0;
uint32_t totalBytesSent  = 0;
uint32_t camErrorCount   = 0;
uint32_t udpErrorCount   = 0;
uint32_t mdnsFailCount   = 0;

uint32_t framesInWindow    = 0;
uint32_t bytesInWindow     = 0;
uint32_t lastStatsTime     = 0;
uint32_t lastMdnsRetryTime = 0;
bool     isWifiConnected   = false;

// Forward declaration
void updateMotorOutputs();
void pollControlPacket();

IPAddress resolveServerIP() {
  if (cachedServerIP != IPAddress(0, 0, 0, 0)) return cachedServerIP;

  if (millis() - lastMdnsRetryTime < 2000) return cachedServerIP;
  lastMdnsRetryTime = millis();

  Serial.println("[ESP-DIAG] Resolving mDNS target host...");
  IPAddress ip = MDNS.queryHost(TARGET_HOST);
  if (ip == IPAddress(0, 0, 0, 0)) {
    ip = MDNS.queryHost("larvtop.local");
  }

  if (ip != IPAddress(0, 0, 0, 0)) {
    cachedServerIP = ip;
    Serial.printf("[ESP-DIAG] Target IP resolved successfully: %s\n", cachedServerIP.toString().c_str());
  } else {
    mdnsFailCount++;
    Serial.printf("[ESP-DIAG] mDNS resolution failed for '%s' (Failures: %u)\n", TARGET_HOST, mdnsFailCount);
  }
  return cachedServerIP;
}

int fetchFrameFromCamera(uint8_t* buffer, size_t maxLen) {
  while (Serial1.available()) {
    Serial1.read();
  }

  Serial1.println("CAPTURE");

  unsigned long startTime = millis();
  String header = "";

  // Hunt for the "SIZE:" header across incoming lines for up to 50ms, skipping empty/stray logs
  while (millis() - startTime < 50) {
    if (Serial1.available()) {
      header = Serial1.readStringUntil('\n');
      header.trim();

      if (header.startsWith("SIZE:")) {
        break; // Successfully found our target header!
      }
    }
  }

  if (!header.startsWith("SIZE:")) {
    camErrorCount++;
    return -1;
  }

  int imgSize = header.substring(5).toInt();
  if (imgSize <= 0 || (size_t)imgSize > maxLen) {
    camErrorCount++;
    Serial.printf("[ESP-CAM ERROR] Out of bounds image size: %d bytes (Cam Errors: %u)\n", imgSize, camErrorCount);
    return -1;
  }

  size_t bytesRead = Serial1.readBytes((char*)buffer, imgSize);
  if (bytesRead != (size_t)imgSize) {
    camErrorCount++;
    Serial.printf("[ESP-CAM ERROR] Partial read mismatch. Expected: %d, Got: %u (Cam Errors: %u)\n", imgSize, bytesRead, camErrorCount);
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
    int lightState = 0;

    if (sscanf(packetBuffer, "%f,%f,%c,%d", &v1, &v2, &modeChar, &lightState) >= 3) {
      currentVal1 = v1;
      currentVal2 = v2;

      if (modeChar == 'I') {
        isContinuousMode = false;
      } else if (modeChar == 'C') {
        isContinuousMode = true;
      }
      frameRequested = true;
      updateMotorOutputs();

      if (lightState != lastSentLightState) {
        lastSentLightState = lightState;
        if (lightState == 1) {
          Serial1.println("LIGHT_ON");
        } else {
          Serial1.println("LIGHT_OFF");
        }
      }
    }
  }
}

void updateMotorOutputs() {
  float leftVal  = currentVal1;
  float rightVal = -currentVal2;

  int leftPWM  = (int)(abs(leftVal) * 255.0f);
  int rightPWM = (int)(abs(rightVal) * 255.0f);

  leftPWM  = constrain(leftPWM, 0, 255);
  rightPWM = constrain(rightPWM, 0, 255);

  if (leftVal > 0.05f) {
    analogWrite(LEFT_MOTOR_FWD, leftPWM);
    analogWrite(LEFT_MOTOR_BACK, 0);
  } else if (leftVal < -0.05f) {
    analogWrite(LEFT_MOTOR_FWD, 0);
    analogWrite(LEFT_MOTOR_BACK, leftPWM);
  } else {
    analogWrite(LEFT_MOTOR_FWD, 0);
    analogWrite(LEFT_MOTOR_BACK, 0);
  }

  if (rightVal > 0.05f) {
    analogWrite(RIGHT_MOTOR_FWD, rightPWM);
    analogWrite(RIGHT_MOTOR_BACK, 0);
  } else if (rightVal < -0.05f) {
    analogWrite(RIGHT_MOTOR_FWD, 0);
    analogWrite(RIGHT_MOTOR_BACK, rightPWM);
  } else {
    analogWrite(RIGHT_MOTOR_FWD, 0);
    analogWrite(RIGHT_MOTOR_BACK, 0);
  }
}

void sendFrameUDP(IPAddress serverIP, int imgBytes, uint32_t captureTimeUs, uint32_t& txTimeUsOut) {
  static uint8_t frameId = 0;
  static uint8_t counter = 0;
  frameId++;
  counter++;

  uint32_t txStart = micros();
  uint16_t totalLen = imgBytes;
  uint8_t totalChunks = (totalLen + UDP_CHUNK_SIZE - 1) / UDP_CHUNK_SIZE;
  bool frameHadError = false;

  for (uint8_t chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    pollControlPacket();
    updateMotorOutputs();

    size_t offset = chunkIdx * UDP_CHUNK_SIZE;
    size_t bytesLeft = totalLen - offset;
    size_t chunkSize = (bytesLeft < UDP_CHUNK_SIZE) ? bytesLeft : UDP_CHUNK_SIZE;

    if (!udp.beginPacket(serverIP, TARGET_PORT)) {
      udpErrorCount++;
      frameHadError = true;
      Serial.printf("[ESP-UDP ERROR] beginPacket failed! (UDP Errors: %u)\n", udpErrorCount);
      break;
    }

    chunkTxBuffer[0] = frameId;
    chunkTxBuffer[1] = chunkIdx;
    chunkTxBuffer[2] = totalChunks;
    chunkTxBuffer[3] = counter;

    memcpy(chunkTxBuffer + 4, &captureTimeUs, 4);

    uint32_t currentElapsedTx = micros() - txStart;
    memcpy(chunkTxBuffer + 8, &currentElapsedTx, 4);

    memcpy(chunkTxBuffer + UDP_HEADER_SIZE, payloadBuffer + offset, chunkSize);
    size_t written = udp.write(chunkTxBuffer, chunkSize + UDP_HEADER_SIZE);
    if (written != chunkSize + UDP_HEADER_SIZE) {
      Serial.printf("[ESP-UDP WARN] Written bytes (%u) != expected (%u)\n", written, chunkSize + UDP_HEADER_SIZE);
    }

    if (!udp.endPacket()) {
      udpErrorCount++;
      frameHadError = true;
      Serial.printf("[ESP-UDP ERROR] endPacket failed! (UDP Errors: %u)\n", udpErrorCount);
      break;
    }

    delayMicroseconds(50);
  }

  txTimeUsOut = micros() - txStart;

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

    Serial.println("\n------------- [ESP32 COMPREHENSIVE DIAGNOSTICS] -------------");
    Serial.printf("  FPS:                %.1f frames/sec\n", fps);
    Serial.printf("  Streaming Mode:     %s\n", isContinuousMode ? "CONTINUOUS" : "LOCKSTEP");
    Serial.printf("  Data Throughput:    %.2f KB/s\n", kbps);
    Serial.printf("  Avg Frame Size:     %.1f KB\n", avgFrameKB);
    Serial.printf("  Wi-Fi Status:       %s | RSSI: %d dBm | SSID: %s\n",
                  (WiFi.status() == WL_CONNECTED) ? "ONLINE" : "OFFLINE",
                  WiFi.RSSI(), WiFi.SSID().c_str());
    Serial.printf("  Target Server:      %s:%d\n", cachedServerIP.toString().c_str(), TARGET_PORT);
    Serial.printf("  Error Counters:     Cam Errors: %u | UDP Errors: %u | mDNS Failures: %u\n",
                  camErrorCount, udpErrorCount, mdnsFailCount);
    Serial.printf("  Total Sent:         %u frames (%u KB)\n", totalFramesSent, totalBytesSent / 1024);
    Serial.println("-------------------------------------------------------------");

    framesInWindow = 0;
    bytesInWindow  = 0;
    lastStatsTime  = now;
  }
}

void setup() {
  delay(1000);
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  Serial.println("\n[ESP-BOOT] Initializing ESP32 Steering & Streaming Node...");

  Serial1.setRxBufferSize(16384);
  Serial1.setTimeout(30);
  Serial1.begin(1000000, SERIAL_8N1, D0, D1);

  pinMode(RIGHT_MOTOR_FWD, OUTPUT);
  pinMode(RIGHT_MOTOR_BACK, OUTPUT);
  pinMode(LEFT_MOTOR_FWD, OUTPUT);
  pinMode(LEFT_MOTOR_BACK, OUTPUT);

  analogWrite(RIGHT_MOTOR_FWD, 0);
  analogWrite(RIGHT_MOTOR_BACK, 0);
  analogWrite(LEFT_MOTOR_FWD, 0);
  analogWrite(LEFT_MOTOR_BACK, 0);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  for (size_t i = 0; i < numNetworks; i++) {
    wifiMulti.addAP(wifiNetworks[i].ssid, wifiNetworks[i].password);
  }

  Serial.print("[ESP-WIFI] Connecting to Wi-Fi multi networks...");
  while (wifiMulti.run() != WL_CONNECTED) {
    Serial.print(".");
    delay(200);
  }
  Serial.println(" CONNECTED!");
  Serial.printf("[ESP-WIFI] IP Address: %s | RSSI: %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());

  isWifiConnected = true;

  MDNS.begin("nano-esp32");
  udp.begin(CONTROL_PORT);

  lastStatsTime = millis();
  Serial.println("[ESP-BOOT] Setup completed successfully. Entering main loop.");
}

void loop() {
  if (wifiMulti.run() != WL_CONNECTED) {
    if (isWifiConnected) {
      isWifiConnected = false;
      cachedServerIP = IPAddress(0, 0, 0, 0);
      currentVal1 = 0.0f;
      currentVal2 = 0.0f;
      updateMotorOutputs();
      Serial.println("[ESP-WIFI WARNING] Wi-Fi connection lost!");
    }
    delay(20);
    return;
  } else if (!isWifiConnected) {
    isWifiConnected = true;
    frameRequested = true;
    Serial.println("[ESP-WIFI] Wi-Fi connection re-established.");
  }

  IPAddress serverIP = resolveServerIP();
  if (serverIP == IPAddress(0, 0, 0, 0)) {
    delay(50);
    return;
  }

  pollControlPacket();
  updateMotorOutputs();

  if (isContinuousMode || frameRequested) {
    totalFramesAttempted++;
    uint32_t tCaptureStart = micros();
    int imgBytes = fetchFrameFromCamera(payloadBuffer, MAX_PAYLOAD_SIZE);
    uint32_t captureTimeUs = micros() - tCaptureStart;

    if (imgBytes > 0) {
      uint32_t txTimeUs = 0;
      sendFrameUDP(serverIP, imgBytes, captureTimeUs, txTimeUs);
    } else {
      if (totalFramesAttempted % 20 == 0) {
        Serial.printf("[ESP-DIAG WARN] Failed to fetch frame from camera (Attempt #%u, Cam Errors: %u)\n", totalFramesAttempted, camErrorCount);
      }
    }
    frameRequested = false;
  }

  printRecurringStats();
  updateMotorOutputs();
  yield();
}
