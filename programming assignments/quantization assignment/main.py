import os, time, pathlib, numpy as np, tensorflow as tf
from tensorflow import keras

# set up baseline model, display summary
baseline = tf.keras.models.load_model("artifacts/resnet101v2_baseline.keras")
baseline.summary()
# time.sleep(10)

baseline.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

num_inferences = 100  # Number of times to run inference for averaging
latencies = []

for _ in range(num_inferences):
    start_time = time.perf_counter()
    _ = baseline.predict(dummy_input)
    end_time = time.perf_counter()
    latencies.append(end_time - start_time)

average_latency = np.mean(latencies)
std_latency = np.std(latencies)

print(f"Average inference latency: {average_latency:.4f} seconds")
print(f"Standard deviation of latency: {std_latency:.4f} seconds")

image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
image_path = tf.keras.utils.get_file('grace_hopper.jpg', image_url)

# 3. Preprocess Image for ResNet (224x224)
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.resnet_v2.preprocess_input(x)

# --- BASELINE PREDICTION ---
start = time.time()
baseline_preds = baseline.predict(x)
print(f"Time: {time.time() - start:.4f}s")
print("Top 3 Predictions:", tf.keras.applications.resnet_v2.decode_predictions(baseline_preds, top=3)[0])
print("----------------------------------------------------------")

# -------------------------------------------------------------------------------------------- #

# set up quantized model
quantized = tf.lite.Interpreter(model_path="artifacts/resnet101v2_int8_dynamic.tflite")
quantized.allocate_tensors()

input_details = quantized.get_input_details()
output_details = quantized.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']
input_dtype = input_details[0]['dtype']
model_h = input_details[0]['shape'][1]
model_w = input_details[0]['shape'][2]

img = tf.keras.preprocessing.image.load_img(image_path, target_size=(model_h, model_w))
x = tf.keras.preprocessing.image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = tf.keras.applications.resnet_v2.preprocess_input(x)

input_data = x.astype(input_dtype)

print("Running inference...")
quantized.set_tensor(input_index, input_data)

start_time = time.perf_counter()
quantized.invoke()
end_time = time.perf_counter()

output_data = quantized.get_tensor(output_index)

preds = tf.keras.applications.resnet_v2.decode_predictions(output_data, top=3)[0]

print("\n--- Results ---")
print(f"Inference Latency: {(end_time - start_time) * 1000:.2f} ms")
print(f"Top Prediction: {preds[0][1]} ({preds[0][2]*100:.2f}%)")
print("-" * 20)
print("Top 3 Classes:")
for i, (id, label, prob) in enumerate(preds):
    print(f"{i+1}. {label}: {prob*100:.2f}%")
