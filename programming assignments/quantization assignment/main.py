import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def load_and_preprocess_image(url, target_size):
    """Downloads from GitHub Raw and preprocesses for ResNetV2."""
    fname = str(abs(hash(url))) + ".jpg"
    try:
        image_path = tf.keras.utils.get_file(fname, url)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) 
        return x
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

def check_top_k(preds, true_idx, k=3):
    """Checks if the true label index is within the top K predictions."""
    top_k_indices = np.argsort(preds)[0][-k:][::-1]
    return true_idx in top_k_indices

TEST_SET = [
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG", 1, "Goldfish"),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01484850_great_white_shark.JPEG", 2, "Great White Shark"),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01518878_ostrich.JPEG", 9, "Ostrich"),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02391049_zebra.JPEG", 340, "Zebra"),
    ("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02099601_golden_retriever.JPEG", 207, "Golden Retriever")
]

# Initialize containers for results
baseline_stats = {'top1': 0, 'top3': 0, 'latencies': []}
quant_stats = {'top1': 0, 'top3': 0, 'latencies': []}

total_samples = len(TEST_SET)

# --------------------------------------------------------------------------- #
# 2. Evaluate Baseline Model (Keras .h5/.keras)
# --------------------------------------------------------------------------- #
print("\n--- Loading Baseline Model ---")
try:
    baseline = tf.keras.models.load_model("artifacts/resnet101v2_baseline.keras")
    
    # Warmup
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    baseline.predict(dummy_input, verbose=0)

    print(f"Running evaluation on {total_samples} images...")
    
    for url, true_idx, label_name in TEST_SET:
        # Load Data
        input_data = load_and_preprocess_image(url, (224, 224))
        if input_data is None: continue

        # Inference
        start = time.perf_counter()
        preds = baseline.predict(input_data, verbose=0)
        end = time.perf_counter()
        
        baseline_stats['latencies'].append((end - start) * 1000) # ms

        # Accuracy Checks
        probs = preds # .predict returns probabilities already
        pred_idx = np.argmax(probs, axis=1)[0]

        if pred_idx == true_idx:
            baseline_stats['top1'] += 1
        
        if check_top_k(probs, true_idx, k=3):
            baseline_stats['top3'] += 1
        else:
            print(f"  [Baseline Miss] Expected: {label_name} | Got ID: {pred_idx}")

except OSError:
    print("Error: Baseline model file not found in 'artifacts/'. Skipping.")


# --------------------------------------------------------------------------- #
# 3. Evaluate Quantized Model (TFLite)
# --------------------------------------------------------------------------- #
print("\n--- Loading Quantized Model (TFLite) ---")
try:
    quantized = tf.lite.Interpreter(model_path="artifacts/resnet101v2_int8_dynamic.tflite")
    quantized.allocate_tensors()

    input_details = quantized.get_input_details()
    output_details = quantized.get_output_details()
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    h = input_details[0]['shape'][1]
    w = input_details[0]['shape'][2]
    input_dtype = input_details[0]['dtype']

    print(f"Running evaluation on {total_samples} images...")

    for url, true_idx, label_name in TEST_SET:
        input_data = load_and_preprocess_image(url, (h, w))
        if input_data is None: continue

        input_data = input_data.astype(input_dtype)

        quantized.set_tensor(input_index, input_data)
        
        start = time.perf_counter()
        quantized.invoke()
        end = time.perf_counter()
        
        quant_stats['latencies'].append((end - start) * 1000) # ms

        output_data = quantized.get_tensor(output_index)
        
        pred_idx = np.argmax(output_data, axis=1)[0]

        if pred_idx == true_idx:
            quant_stats['top1'] += 1
        
        if check_top_k(output_data, true_idx, k=3):
            quant_stats['top3'] += 1
        else:
            print(f"  [Quantized Miss] Expected: {label_name} | Got ID: {pred_idx}")

except ValueError:
     print("Error: Quantized model file not found in 'artifacts/'. Skipping.")


print("\n" + "="*40)
print("       MODEL COMPARISON RESULTS       ")
print("="*40)

# Helper to print stats
def print_stats(name, stats, total):
    if not stats['latencies']:
        print(f"{name}: No data collected.")
        return
    
    avg_lat = np.mean(stats['latencies'])
    top1_acc = (stats['top1'] / total) * 100
    top3_acc = (stats['top3'] / total) * 100
    
    print(f"Model: {name}")
    print(f"  - Avg Latency:  {avg_lat:.2f} ms")
    print(f"  - Top-1 Acc:    {top1_acc:.1f}% ({stats['top1']}/{total})")
    print(f"  - Top-3 Acc:    {top3_acc:.1f}% ({stats['top3']}/{total})")
    print("-" * 40)

print_stats("Baseline (Keras)", baseline_stats, total_samples)
print_stats("Quantized (TFLite)", quant_stats, total_samples)