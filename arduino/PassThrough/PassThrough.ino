void setup() {
  Serial.begin(115200);   // Kommunikation med datorn via USB
  Serial1.begin(115200);  // Kommunikation med ESP32-CAM (stiften TX1/RX0)
}

void loop() {
  while (Serial.available()) Serial1.write(Serial.read());
  while (Serial1.available()) Serial.write(Serial1.read());
}
