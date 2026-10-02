#include "esp_camera.h"

// Den inbyggda lilla röda lysdioden på ESP32-CAM (GPIO 33, låg = tänd)
#define LED_BUILTIN_CAM 33

unsigned long lastCaptureTime = 0;

void warmupCamera(int frames) {
  for (int i = 0; i < frames; i++) {
    camera_fb_t * dummy = esp_camera_fb_get();
    if (dummy) esp_camera_fb_return(dummy);
    delay(30);
  }
}

void blinkStatus(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN_CAM, LOW);  // Tänd
    delay(delayMs);
    digitalWrite(LED_BUILTIN_CAM, HIGH); // Släck
    delay(delayMs);
  }
}

void setup() {
  // Höjd seriehastighet till 1 MHz för att hinna skicka bilder utan packet loss
  Serial.begin(1000000);

  pinMode(LED_BUILTIN_CAM, OUTPUT);
  digitalWrite(LED_BUILTIN_CAM, HIGH); // Släckt från start

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;   config.pin_d1 = 18;  config.pin_d2 = 19;
  config.pin_d3 = 21;  config.pin_d4 = 36;  config.pin_d5 = 39;
  config.pin_d6 = 34;  config.pin_d7 = 35;  config.pin_xclk = 0;
  config.pin_pclk = 22; config.pin_vsync = 25; config.pin_href = 23;
  config.pin_sscb_sda = 26; config.pin_sscb_scl = 27; config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Optimerat för låg latens (320x240)
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;            // Låg komprimeringstid / lagom storlek
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
    s->set_wb_mode(s, 2); // Inomhus/Cloudy
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
  }

  warmupCamera(5);

  // Blinkar 3 gånger snabbt när allt är redo!
  blinkStatus(3, 100);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "CAPTURE") {
      digitalWrite(LED_BUILTIN_CAM, LOW); // Tänd LED under utläsning

      if (millis() - lastCaptureTime > 3000) {
        warmupCamera(2);
      }

      camera_fb_t * fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("ERR:FB_GET_NULL");
        digitalWrite(LED_BUILTIN_CAM, HIGH);
        return;
      }

      lastCaptureTime = millis();

      // Skicka bildstorlek
      Serial.printf("SIZE:%d\n", fb->len);
      Serial.flush();

      // Skicka råa JPEG-bytes
      Serial.write(fb->buf, fb->len);
      Serial.flush();

      esp_camera_fb_return(fb);
      digitalWrite(LED_BUILTIN_CAM, HIGH); // Släck LED
    }
    else if (cmd.length() > 0) {
      Serial.printf("ERR:UNKNOWN_CMD_%s\n", cmd.c_str());
    }
  }
}
