#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "esp_camera.h"

// ================== CONFIGURACIÓN ==================
const char* ssid = "CABRERA";
const char* password = "00000000";
const char* serverUrl = "https://esp32-person-detector.onrender.com/detect";

#define PIR_PIN 13                    // Pin del sensor PIR
const unsigned long KEEP_ALIVE_INTERVAL = 13 * 60 * 1000;  // 13 minutos

// ================== PINES CÁMARA ==================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ================== BASE64 ==================
static const char b64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_enc_len(int plainLen) { return ((plainLen + 2) / 3) * 4 + 1; }

void b64_encode(char* output, const uint8_t* input, int inputLen) {
  // ... (código base64 igual que antes - lo mantengo igual)
  int i = 0, j = 0;
  uint8_t a3[3], a4[4];
  while (inputLen--) {
    a3[i++] = *input++;
    if (i == 3) {
      a4[0] = (a3[0] & 0xFC) >> 2;
      a4[1] = ((a3[0] & 0x03) << 4) + ((a3[1] & 0xF0) >> 4);
      a4[2] = ((a3[1] & 0x0F) << 2) + ((a3[2] & 0xC0) >> 6);
      a4[3] = a3[2] & 0x3F;
      for (i = 0; i < 4; i++) output[j++] = b64chars[a4[i]];
      i = 0;
    }
  }
  if (i > 0) {
    for (int k = i; k < 3; k++) a3[k] = 0;
    a4[0] = (a3[0] & 0xFC) >> 2; a4[1] = ((a3[0] & 0x03) << 4) + ((a3[1] & 0xF0) >> 4);
    a4[2] = ((a3[1] & 0x0F) << 2) + ((a3[2] & 0xC0) >> 6); a4[3] = a3[2] & 0x3F;
    for (int k = 0; k < i + 1; k++) output[j++] = b64chars[a4[k]];
    while (i++ < 3) output[j++] = '=';
  }
  output[j] = '\0';
}

// ================== FUNCIONES ==================
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM; config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Error inicializando cámara ❌");
    return;
  }
  Serial.println("Cámara inicializada ✅");
}

void captureAndSend() {
  Serial.println("\nCapturando foto...");

  digitalWrite(4, HIGH);
  delay(80);
  camera_fb_t * fb = esp_camera_fb_get();
  digitalWrite(4, LOW);

  if (!fb || fb->len == 0) {
    Serial.println("Error capturando foto ❌");
    if (fb) esp_camera_fb_return(fb);
    return;
  }

  int index = fb->len;
  char* encoded = (char*) malloc(b64_enc_len(index) + 1);
  if (!encoded) {
    Serial.println("Sin memoria para Base64 ❌");
    esp_camera_fb_return(fb);
    return;
  }

  b64_encode(encoded, fb->buf, index);
  esp_camera_fb_return(fb);

  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http2;
  http2.begin(client, serverUrl);
  http2.addHeader("Content-Type", "application/json");
  http2.setTimeout(60000);

  String jsonPayload = "{\"image\":\"" + String(encoded) + "\"}";
  free(encoded);

  int code = http2.POST(jsonPayload);
  Serial.print("Respuesta servidor: ");
  Serial.println(code > 0 ? http2.getString() : http2.errorToString(code));
  http2.end();
}

// ================== SETUP ==================
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(4, OUTPUT); digitalWrite(4, LOW);   // Flash
  pinMode(PIR_PIN, INPUT);                    // PIR

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi conectado ✅");

  initCamera();
  Serial.println("Sistema iniciado. Esperando movimiento o keep-alive...");
}

unsigned long lastSendTime = 0;

void loop() {
  unsigned long now = millis();
  bool motion = digitalRead(PIR_PIN);

  // === MODO MOVIMIENTO: 3 fotos en ráfaga ===
  if (motion) {
    Serial.println("¡MOVIMIENTO DETECTADO! Enviando ráfaga de 3 fotos...");
    for (int i = 0; i < 3; i++) {
      captureAndSend();
      if (i < 2) delay(700);   // Pequeña pausa entre fotos
    }
    lastSendTime = millis();
    delay(3000); // Evita que dispare muchas veces seguidas
  }

  // === MODO KEEP-ALIVE: cada 13 minutos ===
  else if (now - lastSendTime >= KEEP_ALIVE_INTERVAL) {
    Serial.println("Enviando foto keep-alive: \"No te duermas pequeño\"");
    captureAndSend();
    lastSendTime = millis();
  }

  // Comando manual 't' (para pruebas)
  if (Serial.available()) {
    if (Serial.read() == 't') {
      Serial.println("Captura manual...");
      captureAndSend();
      lastSendTime = millis();
    }
  }

  delay(200); // Pequeña pausa para no saturar el loop
}