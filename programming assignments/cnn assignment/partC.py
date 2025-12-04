import tensorflow as tf
import numpy as np
import os
import time

# 1. setup & data Reloading
print("Loading Data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalize (must match Part B exactly)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 2. load the baseline model
model_path = 'model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find {model_path}. Please run Part B and save the model first.")

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

# 3. quantization (C2)
def representative_data_gen():
    # use 300 samples for calibration
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(300):
        yield [input_value]

print("Quantizing model (this may take a moment)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# enforce full INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# save quantized model
tflite_filename = 'model_quant_int8.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model_quant)
print(f"Quantized model saved to {tflite_filename}")

# 4. Benchmarking & Analysis (C3)
def benchmark_tflite(tflite_path, test_images, test_labels):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    scale, zero_point = input_details[0]['quantization']
    is_quantized = input_details[0]['dtype'] == np.uint8
    
    correct_count = 0
    latencies = []
    n = len(test_images)
    
    print(f"Benchmarking {n} images...")
    
    for i in range(n):
        img = test_images[i:i+1]
        
        # quantize input if needed
        if is_quantized:
            img = (img / scale + zero_point).astype(np.uint8)
            
        interpreter.set_tensor(input_details[0]['index'], img)
        
        start = time.time()
        interpreter.invoke()
        end = time.time()
        
        if i >= 10: # skip warm-up
            latencies.append((end - start) * 1000)
            
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == test_labels[i]:
            correct_count += 1
            
    return os.path.getsize(tflite_path) / (1024**2), np.mean(latencies), (correct_count / n) * 100

# run benchmark
# using 2000 samples for speed
q_size, q_lat, q_acc = benchmark_tflite(tflite_filename, x_test[:2000], y_test[:2000])

print(f"\n--- Results Table Data ---")
print(f"Model:           INT8 PTQ")
print(f"File Size:       {q_size:.4f} MB")
print(f"Latency (CPU):   {q_lat:.4f} ms/img")
print(f"Test Top-1:      {q_acc:.2f}%")