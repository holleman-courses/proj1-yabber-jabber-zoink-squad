#include <Wire.h>
#include <Arduino.h>
#include <TensorFlowLite.h>
#include <TinyMLShield.h>  
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "trained_model.h"

// Hardware Configuration
#define LED_PIN 13
static bool is_initialized = false;

// Image Configuration
#define CAMERA_WIDTH 176     // QCIF width
#define CAMERA_HEIGHT 144    // QCIF height
#define RESIZED_WIDTH 96
#define RESIZED_HEIGHT 96
#define DETECTION_THRESHOLD .49f

// Tensor Arena
constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize]; 

// Image Buffer
int8_t image_data[RESIZED_WIDTH * RESIZED_HEIGHT];
bool led_state = false;  // For tracking LED toggle state

// TFLite Globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Quantization parameters
float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

void setup() {
  // Initialize hardware
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }

  Serial.begin(115200);
  while(!Serial) {
    delay(10);
  }

  // Initialize camera
  if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
    Serial.println("Failed to initialize camera!");
    while(1);
  }
  
  setupTensorFlow();

  // Blink LEDs to indicate ready
  for (int i = 0; i < 6; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
  }
  
  Serial.println("System Ready");
  Serial.println("Commands:");
  Serial.println("c - Capture and process image");
  Serial.println("v - View last captured image data");
  Serial.println("r - Toggle status LED");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    
    if (cmd == 'c') {
      digitalWrite(LED_PIN, HIGH);
      if (captureAndProcessImage()) {
        runInference();
        outputResults();
      }
      digitalWrite(LED_PIN, LOW);
    }
    else if (cmd == 'v') {
      // View last captured image data
      Serial.println("Last captured image data (96x96):");
      for (int y = 0; y < RESIZED_HEIGHT; y++) {
        for (int x = 0; x < RESIZED_WIDTH; x++) {
          Serial.print(image_data[y * RESIZED_WIDTH + x]);
          Serial.print(" ");
        }
        Serial.println();
      }
    }
    else if (cmd == 'r') {
      // Toggle detection LED
      led_state = !led_state;
      digitalWrite(LED_PIN, led_state ? HIGH : LOW);
      Serial.print("Detection LED ");
      Serial.println(led_state ? "ON" : "OFF");
    }
  }
  delay(100);
}

bool captureAndProcessImage() {
  byte camera_data[CAMERA_WIDTH * CAMERA_HEIGHT]; // QCIF grayscale buffer
  
  // Read camera data
  Camera.readFrame(camera_data);

  // Crop center 96x96 from the 176x144 image
  int min_x = (176 - 96) / 2;
  int min_y = (144 - 96) / 2;
  int index = 0;

  // Crop 96x96 image. 
  for (int y = min_y; y < min_y + 96; y++) {
    for (int x = min_x; x < min_x + 96; x++) {
      image_data[index++] = static_cast<int8_t>(camera_data[(y * 176) + x] - 128); // convert TF input image to signed 8-bit
    }
  }
  Serial.println();
  // Copy to model input tensor
  for (int i = 0; i < RESIZED_WIDTH * RESIZED_HEIGHT; i++) {
    input->data.int8[i] = image_data[i];
  }
  Serial.println("Model input values:");
  for(int i=0; i<8; i++) {
    Serial.print(input->data.int8[i]);
    Serial.print(" ");
  }
  Serial.println();
  Serial.println("Image captured and processed");
  digitalWrite(LEDB, LOW);
  delay(2000);
  digitalWrite(LEDB, HIGH);
  delay(100);
  return true;
}

void setupTensorFlow() {
  model = tflite::GetModel(trained_model);
  
  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Get quantization parameters
  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.println("TFLite Initialized");
}

void runInference() {
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    while(1);
  }
}

void outputResults() {
  Serial.print("Model output: ");
  Serial.println((output->data.int8[0] - output_zero_point) * output_scale);
  float prediction = (output->data.int8[0] - output_zero_point) * output_scale;
  if (prediction > DETECTION_THRESHOLD) {
    Serial.println("That's probably an airplane!");
    digitalWrite(LEDR, HIGH);
    delay(2000);
    digitalWrite(LEDG, LOW);
  } else {
    Serial.println("That's not an airplane! :(");
    digitalWrite(LEDG, HIGH);
    delay(2000);
    digitalWrite(LEDR, LOW);
  }
  delay(1000);
}