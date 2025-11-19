# MobileNetV2 → FP32 .keras → TFLite INT8 (dynamic range) → latency
import os, time, pathlib, numpy as np, tensorflow as tf
from tensorflow import keras

outdir = pathlib.Path("artifacts"); outdir.mkdir(exist_ok=True)

# 1) Load compact pretrained model (swap to EfficientNetB0 in the note below)
model = keras.applications.ResNet101V2(weights="imagenet", include_top=True, input_shape=(224,224,3))
preprocess = keras.applications.mobilenet_v2.preprocess_input

# 2) Save FP32 baseline
baseline_path = outdir / "resnet101v2_baseline.keras"
model.save(baseline_path.as_posix())
print("Baseline size (MB):", round(baseline_path.stat().st_size / (1024*1024), 2))

# 3) Convert to TFLite (dynamic range INT8)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range
tflite_model = converter.convert()
tflite_path = outdir / "resnet101v2_int8_dynamic.tflite"
tflite_path.write_bytes(tflite_model)
print("TFLite INT8 (dynamic) size (MB):", round(tflite_path.stat().st_size / (1024*1024), 2))

# 4) Build an input (toggle USE_IMAGE to test with a real image)
USE_IMAGE = False
if USE_IMAGE:
    from PIL import Image
    img = Image.open("../wit_help.jpg").convert("RGB").resize((224,224))  # <-- replace path
    x = np.array(img).astype("float32")
    x = preprocess(x)[None, ...]  # add batch dim
else:
	x = np.random.randint(0, 256, (1,224,224,3)).astype("float32")
	x = preprocess(x)

# 5) Measure latency with the LiteRT/TFLite Interpreter (note dtype cast)
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]; out = interpreter.get_output_details()[0]

x = x.astype(inp["dtype"])  # dynamic‑range expects float32 by default
for _ in range(5):  # warmup
    interpreter.set_tensor(inp["index"], x); interpreter.invoke()

times = []
for _ in range(50):
    t0 = time.time()
    interpreter.set_tensor(inp["index"], x); interpreter.invoke()
    _ = interpreter.get_tensor(out["index"])
    times.append((time.time() - t0) * 1000)

times = np.array(times)
print(f"Latency (ms): mean={times.mean():.2f}, p50={np.percentile(times,50):.2f}, p95={np.percentile(times,95):.2f}")