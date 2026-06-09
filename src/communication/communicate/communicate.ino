void setup() {
  Serial.begin(9600);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    String message = Serial.readStringUntil('\n');

    if (message == "ON") {
        digitalWrite(8, HIGH);
        digitalWrite(9, HIGH);
        digitalWrite(10, HIGH);
        Serial.println("Leds allumées");
    }

    if (message == "OFF") {
        digitalWrite(8, LOW);
        digitalWrite(9, LOW);
        digitalWrite(10, LOW);
        Serial.println("Leds éteintes");
    }
  }
}