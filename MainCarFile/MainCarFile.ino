// Single ESP32: car motors (L298N) + 6-DOF robotic arm
// Arduino-ESP32 core 3.x. No extra libraries needed.
//
// One TCP server (port 5000), one connection from Python.
//
// Commands (newline-terminated):
//   M,<left>,<right>   motor command              (no reply)
//   STOP               stop motors                (no reply)
//   PICK               run the pick sequence      -> STARTED, STEP.., DONE (or BUSY)
//   HOME               move arm to home pose      -> OK (or BUSY)
//   ABORT              freeze the arm             -> ABORTED
//   PING                                          -> PONG
//
// POWER: servos must NOT be powered from the ESP32 pin. Use a separate
// 5-6 V supply and connect ALL grounds together (ESP32, L298N, servo supply).

#include <WiFi.h>

// ---------- Wi-Fi settings ----------
const char* WIFI_SSID = "YourWifiName";
const char* WIFI_PASSWORD = "YourWifiPassword";
const uint16_t TCP_PORT = 5000;

WiFiServer server(TCP_PORT);
WiFiClient client;

// =========================================================
// CAR (L298N)  -- pins unchanged
// =========================================================
const int IN1 = 26;
const int IN2 = 27;
const int IN3 = 32;
const int IN4 = 33;
const int ENA = 25;
const int ENB = 14;

const int PWM_FREQ = 1000;
const int MAX_PWM = 200;   // must match HARD_PWM_LIMIT in Python

const unsigned long COMMAND_TIMEOUT_MS = 700;
unsigned long lastCommandTime = 0;

const unsigned long BRAKE_MS = 150;
bool moving = false;
bool braking = false;
unsigned long brakeStart = 0;

void setMotor(int speed, int pinA, int pinB, int pwmPin) {
  speed = constrain(speed, -MAX_PWM, MAX_PWM);

  if (speed > 0) {
    digitalWrite(pinA, HIGH);
    digitalWrite(pinB, LOW);
  } else if (speed < 0) {
    digitalWrite(pinA, LOW);
    digitalWrite(pinB, HIGH);
  } else {
    digitalWrite(pinA, LOW);
    digitalWrite(pinB, LOW);
  }

  ledcWrite(pwmPin, abs(speed));
}

void stopMotors() {
  setMotor(0, IN1, IN2, ENA);
  setMotor(0, IN3, IN4, ENB);
  moving = false;
  braking = false;
}

void brakeMotors() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENA, 255);
  ledcWrite(ENB, 255);
  braking = true;
  brakeStart = millis();
  moving = false;
}

void haltMotors() {
  if (moving) {
    brakeMotors();
  }
}

void updateBrake() {
  if (braking && millis() - brakeStart > BRAKE_MS) {
    stopMotors();
  }
}

int lastLeft = 0, lastRight = 0;

void applyMotorCommand(const char* command) {
  int leftSpeed, rightSpeed;

  if (sscanf(command, "M,%d,%d", &leftSpeed, &rightSpeed) != 2) return;

  if (leftSpeed != lastLeft || rightSpeed != lastRight) {
    Serial.printf("CMD L=%d R=%d (limit %d)\n", leftSpeed, rightSpeed, MAX_PWM);
    lastLeft = leftSpeed;
    lastRight = rightSpeed;
  }

  if (leftSpeed < -255 || leftSpeed > 255 ||
      rightSpeed < -255 || rightSpeed > 255) {
    haltMotors();
    return;
  }

  if (leftSpeed == 0 && rightSpeed == 0) {
    haltMotors();
  } else {
    braking = false;
    moving = true;
    setMotor(leftSpeed, IN1, IN2, ENA);
    setMotor(rightSpeed, IN3, IN4, ENB);
  }

  lastCommandTime = millis();
}

// =========================================================
// ARM
// =========================================================
enum Joint { BASE, SHOULDER, ELBOW, WRIST, WRIST_ROT, GRIPPER, NUM_JOINTS };

struct ServoCfg {
  const char* name;
  int pin;
  bool enabled;
  bool invert;
  int homeAngle;
};

// Arm pins were MOVED because the old ones clashed with the motor pins
// (14, 25, 26, 27, 33). Rewire the servo signal wires to these GPIOs.
ServoCfg servos[NUM_JOINTS] = {
  // name         pin  enabled invert home
  {"base",        13,  true,   false, 140},  // home is now 100 (was 30)
  {"shoulder",    16,  true,   false, 100},   // always 90
  {"elbow",       17,  true,   false, 90},
  {"wrist",       18,  false,   false, 100},  // always 100
  {"wrist_rot",   19,  false,  false, 90},   // not connected
  {"gripper",     21,  true,  false, 90},   // NOT connected yet
};

const int GRIPPER_OPEN_ANGLE = 30;
const int GRIPPER_CLOSE_ANGLE = 90;

const int SERVO_FREQ = 50;
const int SERVO_BITS = 14;
const int MIN_US = 500;
const int MAX_US = 2500;

const float STEP_DEG = 1.0;
const unsigned long MOVE_INTERVAL_MS = 15;
const unsigned long PAUSE_MS = 1000;

float currentAngle[NUM_JOINTS];
float targetAngle[NUM_JOINTS];
unsigned long lastMoveUpdate = 0;

struct Step {
  const char* name;
  Joint joint;
  int angle;
};

// Base now starts at 100, swings to 30 to drop, and returns to 100.
const Step SEQUENCE[] = {
  {"Gripper open",  GRIPPER, GRIPPER_OPEN_ANGLE},
  {"Elbow to 0",    ELBOW,   0},
  {"Gripper close", GRIPPER, GRIPPER_CLOSE_ANGLE},
  {"Elbow to 90",   ELBOW,   90},
  {"Base to 50",    BASE,    50},
  {"Gripper open",  GRIPPER, GRIPPER_OPEN_ANGLE},
  {"Base to 140",   BASE,    140},
};
const int NUM_STEPS = sizeof(SEQUENCE) / sizeof(SEQUENCE[0]);

bool seqRunning = false;
int stepIndex = 0;
bool pausing = false;
unsigned long pauseStart = 0;

void reply(const String& text) {
  Serial.println("-> " + text);

  if (client && client.connected()) {
    client.print(text + "\n");
  }
}

void writeServo(int j, float angle) {
  if (!servos[j].enabled) return;

  angle = constrain(angle, 0.0f, 180.0f);

  if (servos[j].invert) {
    angle = 180.0f - angle;
  }

  float us = MIN_US + (MAX_US - MIN_US) * angle / 180.0f;
  uint32_t duty = (uint32_t)(us * ((1 << SERVO_BITS) - 1) / 20000.0f);

  ledcWrite(servos[j].pin, duty);
}

bool allAtTarget() {
  for (int j = 0; j < NUM_JOINTS; j++) {
    if (servos[j].enabled && currentAngle[j] != targetAngle[j]) {
      return false;
    }
  }

  return true;
}

void updateMotion() {
  unsigned long now = millis();

  if (now - lastMoveUpdate < MOVE_INTERVAL_MS) return;
  lastMoveUpdate = now;

  for (int j = 0; j < NUM_JOINTS; j++) {
    if (!servos[j].enabled) continue;

    float diff = targetAngle[j] - currentAngle[j];

    if (diff == 0) continue;

    if (fabs(diff) <= STEP_DEG) {
      currentAngle[j] = targetAngle[j];
    } else {
      currentAngle[j] += (diff > 0 ? STEP_DEG : -STEP_DEG);
    }

    writeServo(j, currentAngle[j]);
  }
}

void goHome() {
  for (int j = 0; j < NUM_JOINTS; j++) {
    targetAngle[j] = servos[j].homeAngle;
  }
}

void finishSequence() {
  seqRunning = false;
  pausing = false;
  reply("DONE");
}

void enterStep() {
  while (stepIndex < NUM_STEPS && !servos[SEQUENCE[stepIndex].joint].enabled) {
    reply(String("STEP ") + (stepIndex + 1) + "/" + NUM_STEPS + " " +
          SEQUENCE[stepIndex].name + " (skipped, not connected)");
    stepIndex++;
  }

  if (stepIndex >= NUM_STEPS) {
    finishSequence();
    return;
  }

  const Step& s = SEQUENCE[stepIndex];
  targetAngle[s.joint] = s.angle;
  pausing = false;

  reply(String("STEP ") + (stepIndex + 1) + "/" + NUM_STEPS + " " + s.name);
}

void startSequence() {
  seqRunning = true;
  stepIndex = 0;
  pausing = false;

  targetAngle[SHOULDER] = 100;
  targetAngle[WRIST] = 100;

  reply("STARTED");
  enterStep();
}

void updateSequence() {
  if (!seqRunning) return;

  if (pausing) {
    if (millis() - pauseStart >= PAUSE_MS) {
      stepIndex++;
      enterStep();
    }

    return;
  }

  if (allAtTarget()) {
    if (stepIndex + 1 >= NUM_STEPS) {
      finishSequence();
    } else {
      pausing = true;
      pauseStart = millis();
    }
  }
}

// =========================================================
// COMMAND DISPATCH
// =========================================================
void applyCommand(const char* command) {
  if (command[0] == 'M' && command[1] == ',') {
    applyMotorCommand(command);
    return;
  }

  Serial.printf("<- %s\n", command);

  if (strcmp(command, "STOP") == 0) {
    haltMotors();
    lastCommandTime = millis();
  } else if (strcmp(command, "PICK") == 0) {
    if (seqRunning) reply("BUSY");
    else startSequence();
  } else if (strcmp(command, "HOME") == 0) {
    if (seqRunning) {
      reply("BUSY");
    } else {
      goHome();
      reply("OK");
    }
  } else if (strcmp(command, "ABORT") == 0) {
    seqRunning = false;
    pausing = false;

    for (int j = 0; j < NUM_JOINTS; j++) {
      targetAngle[j] = currentAngle[j];
    }

    reply("ABORTED");
  } else if (strcmp(command, "PING") == 0) {
    reply("PONG");
  } else {
    reply("ERR unknown command");
  }
}

// =========================================================
// SETUP / LOOP
// =========================================================
char buffer[64];
int bufferIndex = 0;

void setup() {
  Serial.begin(115200);

  // Motors
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  ledcAttach(ENA, PWM_FREQ, 8);
  ledcAttach(ENB, PWM_FREQ, 8);
  stopMotors();

  // Arm: attach PWM and hold the home pose.
  for (int j = 0; j < NUM_JOINTS; j++) {
    currentAngle[j] = servos[j].homeAngle;
    targetAngle[j] = servos[j].homeAngle;

    if (servos[j].enabled) {
      ledcAttach(servos[j].pin, SERVO_FREQ, SERVO_BITS);
      writeServo(j, currentAngle[j]);
    }
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    stopMotors();
    delay(300);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());   // put this in ESP32_IP in Python

  server.begin();
  server.setNoDelay(true);

  lastCommandTime = millis();

  Serial.printf("TCP server started on port %u\n", TCP_PORT);
}

void loop() {
  // Accept a new connection.
  if (!client || !client.connected()) {
    if (client) {
      client.stop();
    }

    haltMotors();
    bufferIndex = 0;

    WiFiClient incoming = server.available();

    if (incoming) {
      client = incoming;
      client.setNoDelay(true);
      lastCommandTime = millis();
      Serial.println("Python connected");
      reply("READY");
    }
  }

  // Read newline-terminated commands.
  if (client && client.connected()) {
    while (client.available()) {
      char c = client.read();

      if (c == '\n') {
        buffer[bufferIndex] = '\0';

        if (bufferIndex > 0) {
          applyCommand(buffer);
        }

        bufferIndex = 0;
      } else if (c != '\r') {
        if (bufferIndex < (int)sizeof(buffer) - 1) {
          buffer[bufferIndex++] = c;
        } else {
          haltMotors();
          bufferIndex = 0;
        }
      }
    }
  }

  // Fail-safe: stop motors when commands stop arriving.
  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    haltMotors();
  }

  if (WiFi.status() != WL_CONNECTED) {
    haltMotors();
  }

  updateBrake();
  updateSequence();
  updateMotion();

  delay(2);
}
