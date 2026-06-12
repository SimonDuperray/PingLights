const int LED_PINS[6] = {2, 3, 4, 5, 6, 7};

void setup() {
  Serial.begin(115200);
  for (int i=0; i<6; i++) {
    pinMode(LED_PINS[i], OUTPUT);
    digitalWrite(LED_PINS[i], LOW);
  }
}

void loop() {
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n');
    message.trim();
    Serial.println(message);

    if (message == "OFF") {
      for (int i=0; i < 6; i++) {
        digitalWrite(LED_PINS[i], LOW);
      }
    } else if (message.startsWith("Z")) {
      int zone = message.substring(1).toInt();
      if (zone >= 1 && zone <= 6) {
        for (int i=0; i < 6; i++) {
          digitalWrite(LED_PINS[i], LOW);
        }
        digitalWrite(LED_PINS[zone-1], HIGH);
      }
    }
  }
}