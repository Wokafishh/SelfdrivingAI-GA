#include <WiFi.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>

// --- Hardcoded Network Credentials ---
const char* SEARCH_SSID   = "TP-Link_0958";
const char* WIFI_PASSWORD = "95704883";

// Target: Arch Linux Host
const char* TARGET_HOST = "larvtop";
const int   TARGET_PORT = 8000;

const int LED_PIN_1 = A0;
const int LED_PIN_2 = A1;

bool containsIgnoreCase(String haystack, String needle) {
  haystack.toLowerCase();
  needle.toLowerCase();
  return haystack.indexOf(needle) >= 0;
}

String findBestSSIDMatch(const char* target) {
  Serial.println("\n=== Scanning Nearby Wi-Fi Networks ===");
  int n = WiFi.scanNetworks();
  if (n == 0) {
    Serial.println("[SCAN] No 2.4GHz networks found!");
    return "";
  }

  Serial.printf("[SCAN] Found %d networks:\n", n);
  int bestIndex = -1;
  int bestRSSI = -1000;
  String targetStr = String(target);

  for (int i = 0; i < n; ++i) {
    String scannedSSID = WiFi.SSID(i);
    int rssi = WiFi.RSSI(i);
    Serial.printf("  %d: %s (%d dBm)\n", i + 1, scannedSSID.c_str(), rssi);

    if (containsIgnoreCase(scannedSSID, targetStr)) {
      if (rssi > bestRSSI) {
        bestRSSI = rssi;
        bestIndex = i;
      }
    }
  }

  if (bestIndex != -1) {
    String matched = WiFi.SSID(bestIndex);
    Serial.printf(">>> Matched '%s' to '%s' (%d dBm)\n", target, matched.c_str(), bestRSSI);
    return matched;
  }

  Serial.printf(">>> No match found for '%s'\n", target);
  return "";
}

void fetchInstruction() {
  if (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN_1, LOW);
    digitalWrite(LED_PIN_2, LOW);
    return;
  }

  IPAddress serverIP = MDNS.queryHost(TARGET_HOST);
  if (serverIP == IPAddress(0, 0, 0, 0)) {
    serverIP = MDNS.queryHost("larvtop.local");
  }

  if (serverIP == IPAddress(0, 0, 0, 0)) {
    Serial.println("[mDNS] Could not resolve host.");
    return;
  }

  HTTPClient http;
  String url = "http://" + serverIP.toString() + ":" + String(TARGET_PORT) + "/instruction";

  http.begin(url);
  http.setTimeout(30000); // 30s timeout to allow time to press key
  int httpCode = http.GET();

  if (httpCode > 0) {
    String payload = http.getString();
    payload.trim();

    Serial.printf("[HTTP %d] Received Payload: '%s'\n", httpCode, payload.c_str());

    // Payload format: "XY" where X controls A0 and Y controls A1
    if (payload.length() >= 2) {
      digitalWrite(LED_PIN_1, payload.charAt(0) == '1' ? HIGH : LOW);
      digitalWrite(LED_PIN_2, payload.charAt(1) == '1' ? HIGH : LOW);
    }
  } else {
    Serial.printf("[HTTP Error] %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN_1, OUTPUT);
  pinMode(LED_PIN_2, OUTPUT);
  digitalWrite(LED_PIN_1, LOW);
  digitalWrite(LED_PIN_2, LOW);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  String bestSSID = findBestSSIDMatch(SEARCH_SSID);

  if (bestSSID.length() > 0) {
    Serial.print("Connecting to: ");
    Serial.println(bestSSID);
    WiFi.begin(bestSSID.c_str(), WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[SUCCESS] Wi-Fi Connected!");
      if (MDNS.begin("nano-esp32")) {
        Serial.println("mDNS initialized.");
      }
    }
  }
}

void loop() {
  // Poll Python server for arrow key command every 1 second
  fetchInstruction();
  delay(1000);
}
