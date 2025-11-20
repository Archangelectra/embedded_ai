import os, time, pathlib, numpy as np, tensorflow as tf
from tensorflow import keras

def load_and_preprocess_image(url, target_size):
    """Downloads from GitHub Raw and preprocesses."""
    fname = str(abs(hash(url))) + ".jpg"
    try:
        # GitHub Raw does not require User-Agent headers
        image_path = tf.keras.utils.get_file(fname, url)
        
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) # ResNet V2 (-1 to 1)
        return x
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

# Helper to check if the correct label is in the Top 3 predictions
def check_top_k(preds, true_idx, k=3):
    # argsort returns indices from low to high, so we take last k and reverse them
    top_k_indices = np.argsort(preds)[0][-k:][::-1]
    return true_idx in top_k_indices

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

num_inferences = 100 
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

TEST_SET = [
    # (URL, Class_ID, Label_Name)
    
    # Goldfish (Class 1)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG", 1, "Goldfish"),
    
    # Hammerhead Shark (Class 4)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01484850_great_white_shark.JPEG", 2, "Great White Shark"),
    
    # Ostrich (Class 9)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01518878_ostrich.JPEG", 9, "Ostrich"),
    
    # Zebra (Class 340)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02391049_zebra.JPEG", 340, "Zebra"),
    
    # Golden Retriever (Class 207)
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG", 207, "Golden Retriever")
]

for url, true_label_idx, label_name in TEST_SET:
    input_data = load_and_preprocess_image(url, (224, 224))
    if input_data is None: continue

    start_time = time.perf_counter()
    preds = baseline(input_data, training=False)
    end_time = time.perf_counter()
    
    baseline_stats['latencies'].append((end_time - start_time) * 1000)
    
    # Accuracy Checks
    probs = preds.numpy()
    pred_idx = np.argmax(probs, axis=1)[0]
    
    if pred_idx == true_label_idx:
        baseline_stats['top1'] += 1
    if check_top_k(probs, true_label_idx, k=3):
        baseline_stats['top3'] += 1
    else:
        print(f"  X Baseline Miss: {label_name} (Got ID {pred_idx}")
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