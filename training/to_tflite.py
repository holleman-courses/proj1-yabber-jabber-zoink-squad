import tensorflow as tf
import numpy as np
import subprocess

def representative_data_gen():
    # Replace this with your actual data loading logic
    # This should yield normalized float32 samples in [0,1] range
    num_calibration_steps = 100
    for _ in range(num_calibration_steps):
        # Create dummy input data matching your model's expected input
        dummy_input = np.random.rand(1, 96, 96, 1).astype(np.float32)
        yield [dummy_input]

# Load the trained Keras model
model = tf.keras.models.load_model("trained_model.h5")

# Convert to TFLite with uint8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Set full uint8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Input type
converter.inference_output_type = tf.int8  # Output type

# Convert the model
tflite_model = converter.convert()

# Save the TensorFlow Lite model
model_filename = "trained_model.tflite"
with open(model_filename, "wb") as f:
    f.write(tflite_model)

# Generate C header file
subprocess.run(["xxd", "-i", model_filename, "trained_model.h"])

# Verify the model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== Model Conversion Summary ===")
print(f"Input type: {input_details[0]['dtype']} (int8)")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input scale: {input_details[0]['quantization'][0]}")
print(f"Input zero point: {input_details[0]['quantization'][1]}")
print(f"\nOutput type: {output_details[0]['dtype']} (should be uint8)")
print(f"Output scale: {output_details[0]['quantization'][0]}")
print(f"Output zero point: {output_details[0]['quantization'][1]}")

# Test inference
def run_inference(interpreter, input_data):
    input_data = input_data.astype(np.int8)  # Convert to int8
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])
