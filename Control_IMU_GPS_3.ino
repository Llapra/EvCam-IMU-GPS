#include <Wire.h>
#include <HardwareSerial.h>
#include <TinyGPS++.h>

// --- ZONA DE ALTO RENDIMIENTO ---
#define I2C_CLOCK 600000   // Frecuencia estable para cables de protoboard 
#define BAUDRATE 921600    // Máxima velocidad Serial estable

// --- PINES HARDWARE ---
#define LED_PIN 2
#define SDA_PIN 21
#define SCL_PIN 22
#define RXD2 16            // Cable TX del módulo GPS
#define TXD2 17            // Cable RX del módulo GPS
#define PPS_PIN 4         // Cable PPS del módulo GPS (Importante)

// Direcciones del IMU
const uint8_t MPU_ADDR_1 = 0x68;
const uint8_t MPU_ADDR_2 = 0x69;
uint8_t targetAddr = 0x68;

// --- ESTRUCTURAS BINARIAS (18 BYTES EXACTOS) ---

// Paquete 1: Datos del IMU (18 Bytes)
struct __attribute__((packed)) ImuPacket {
  uint16_t header = 0xBBAA;
  uint32_t esp_micros;
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};
ImuPacket imuData;

// Paquete 2: Pulso PPS Atómico (18 Bytes)
struct __attribute__((packed)) PpsPacket {
  uint16_t header = 0xBBCC;
  uint32_t pps_micros;
  uint8_t padding[12] = {0}; // Relleno para alinear memoria
};
PpsPacket ppsData;

// Paquete 3: Coordenadas y Tiempo Absoluto (18 Bytes exactos)
struct __attribute__((packed)) GpsPacket {
  uint16_t header = 0xBBDD;  // 2 bytes
  float lat;                 // 4 bytes
  float lon;                 // 4 bytes
  uint16_t year;             // 2 bytes
  uint8_t month;             // 1 byte
  uint8_t day;               // 1 byte
  uint8_t hour;              // 1 byte
  uint8_t minute;            // 1 byte
  uint8_t second;            // 1 byte
  uint8_t valid;             // 1 byte
}; // Total: 18 bytes (No requiere padding)
GpsPacket gpsData;

// --- OBJETOS Y VARIABLES GLOBALES ---
TinyGPSPlus gps;
HardwareSerial neogps(2);

volatile bool ppsTriggered = false;
volatile uint32_t ppsMicrosISR = 0;
volatile uint32_t last_pps_time = 0; // NUEVO: Filtro anti-rebote cuántico
unsigned long last_gps_send = 0; // NUEVO: Temporizador para no saturar con coordenadas

// --- INTERRUPCIÓN HARDWARE (Prioridad Máxima) ---
void IRAM_ATTR ppsInterrupt() {
  uint32_t now = micros();
  // Filtro anti-rebote: Ignora perturbaciones en menos de 500ms
  // Garantiza 1 pulso atómico exacto por segundo, sin importar ruido eléctrico
  if (now - last_pps_time > 500000) {
    ppsMicrosISR = now;
    ppsTriggered = true;
    last_pps_time = now;
  }
}

// --- FUNCIONES DE CONTROL I2C ---
void writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void recoverI2C() {
  pinMode(SDA_PIN, INPUT_PULLUP);
  pinMode(SCL_PIN, OUTPUT);
  for (int i = 0; i < 9; i++) {
    digitalWrite(SCL_PIN, LOW); delayMicroseconds(2);
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(2);
  }
  pinMode(SDA_PIN, OUTPUT);
  digitalWrite(SDA_PIN, LOW);
  digitalWrite(SCL_PIN, HIGH); delayMicroseconds(2);
  digitalWrite(SDA_PIN, HIGH);
  pinMode(SDA_PIN, INPUT);
  pinMode(SCL_PIN, INPUT);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  recoverI2C(); 

  // Iniciar comunicaciones
  Serial.begin(BAUDRATE); 
  neogps.begin(9600, SERIAL_8N1, RXD2, TXD2);

// Configurar interrupción del PPS
  // El pin 34 no tiene pull-down físico. Leemos estado crudo en CUALQUIER cambio de voltaje (CHANGE)
  pinMode(PPS_PIN, INPUT); 
  attachInterrupt(digitalPinToInterrupt(PPS_PIN), ppsInterrupt, CHANGE);

  while (!Serial) delay(10);

  // Iniciar I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_CLOCK);

  // 1. Dar tiempo cuántico para que el chip MPU6050 despierte físicamente
  delay(200);

  // 2. Buscar el IMU en el bus
  bool found = false;
  Wire.beginTransmission(MPU_ADDR_1);
  if (Wire.endTransmission() == 0) { targetAddr = MPU_ADDR_1; found = true; }
  else {
    Wire.beginTransmission(MPU_ADDR_2);
    if (Wire.endTransmission() == 0) { targetAddr = MPU_ADDR_2; found = true; }
  }

  // 3. Advertencia si no responde
  if (!found) {
    for (int i = 0; i < 20; i++) { 
       digitalWrite(LED_PIN, !digitalRead(LED_PIN));
       delay(50);
    }
  }

  // 4. Configurar IMU SIEMPRE (Sin el "else" para forzar el encendido)
  writeReg(targetAddr, 0x6B, 0x80); delay(100); // Reset
  writeReg(targetAddr, 0x6B, 0x01);             // Reloj PLL
  writeReg(targetAddr, 0x1A, 0x00);             // Filtro paso bajo OFF
  writeReg(targetAddr, 0x1B, 0x08);             // Giroscopio 500dps
  writeReg(targetAddr, 0x1C, 0x10);             // Acelerómetro 8g
  writeReg(targetAddr, 0x19, 0x00);             // Divisor de tasa = 0

  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // 1. Drenar buffer NMEA del GPS (Sin restricción, previene desbordamiento)
  while (neogps.available() > 0) {
    gps.encode(neogps.read());
  }

// 2. ¿Ocurrió un pulso PPS por hardware?
  if (ppsTriggered) {
    ppsData.pps_micros = ppsMicrosISR; // Tiempo cristalizado en la interrupción
    Serial.write((uint8_t*)&ppsData, sizeof(ppsData));
    // Eliminamos el flush() porque paralizaba el procesador y colgaba el bus I2C
    ppsTriggered = false; // Reset de la bandera
  }

  // 3. Envío de Coordenadas y Tiempo Absoluto (1 vez por segundo)
  if (millis() - last_gps_send > 1000) {
    last_gps_send = millis();
    
    if (gps.location.isValid()) {
      gpsData.lat = gps.location.lat();
      gpsData.lon = gps.location.lng();
    } else {
      gpsData.lat = 0.0;
      gpsData.lon = 0.0;
    }

    // Inyectamos la información del Sello Raíz
    gpsData.year = gps.date.year();
    gpsData.month = gps.date.month();
    gpsData.day = gps.date.day();
    gpsData.hour = gps.time.hour();
    gpsData.minute = gps.time.minute();
    gpsData.second = gps.time.second();
    
    // Validamos que el satélite realmente tenga una época razonable
    gpsData.valid = (gps.time.isValid() && gps.date.isValid() && gps.date.year() > 2020) ? 1 : 0;

    Serial.write((uint8_t*)&gpsData, sizeof(gpsData));
  }

  // 4. Lectura Burst del IMU (Alta velocidad)
  Wire.beginTransmission(targetAddr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  
  if (Wire.requestFrom(targetAddr, (uint8_t)14) == 14) {
      imuData.esp_micros = micros();
      imuData.ax = (Wire.read() << 8 | Wire.read());
      imuData.ay = (Wire.read() << 8 | Wire.read());
      imuData.az = (Wire.read() << 8 | Wire.read());
      Wire.read(); Wire.read(); // Saltar temperatura
      imuData.gx = (Wire.read() << 8 | Wire.read());
      imuData.gy = (Wire.read() << 8 | Wire.read());
      imuData.gz = (Wire.read() << 8 | Wire.read());
    
      // Disparamos el paquete IMU al Python
      Serial.write((uint8_t*)&imuData, sizeof(imuData));
  }
}