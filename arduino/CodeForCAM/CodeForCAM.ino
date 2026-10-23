#include "esp_camera.h"

#define LED_BUILTIN_CAM 33
#define FLASH_LED_PIN 4

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
    digitalWrite(LED_BUILTIN_CAM, LOW);
    delay(delayMs);
    digitalWrite(LED_BUILTIN_CAM, HIGH);
    delay(delayMs);
  }
}

void setup() {
  // Locked to stable 1 Mbps
  Serial.begin(1000000);

  pinMode(LED_BUILTIN_CAM, OUTPUT);
  digitalWrite(LED_BUILTIN_CAM, HIGH);

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;   config.pin_d1 = 18;  config.pin_d2 = 19;
  config.pin_d3 = 21;  config.pin_d4 = 36;  config.pin_d5 = 39;
  config.pin_d6 = 34;  config.pin_d7 = 35;  config.pin_xclk = 0;
  config.pin_pclk = 22; config.pin_vsync = 25; config.pin_href = 23;
  config.pin_sscb_sda = 26; config.pin_sscb_scl = 27; config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // QQVGA (160x120) or FRAMESIZE_96X96. QQVGA is extremely stable for JPEGs and produces tiny ~2KB files.
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("ERR:CAM_INIT_FAILED_0x%x\n", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 3);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
  }

  warmupCamera(5);
  blinkStatus(3, 100);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "CAPTURE") {
      digitalWrite(LED_BUILTIN_CAM, LOW);

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

      Serial.printf("SIZE:%d\n", fb->len);
      Serial.flush();

      Serial.write(fb->buf, fb->len);
      Serial.flush();

      esp_camera_fb_return(fb);
      digitalWrite(LED_BUILTIN_CAM, HIGH);
    }
    else if (cmd == "LIGHT_ON") {
      digitalWrite(FLASH_LED_PIN, HIGH);
    }
    else if (cmd == "LIGHT_OFF") {
      digitalWrite(FLASH_LED_PIN, LOW);
    }
  }
}
