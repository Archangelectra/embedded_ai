# Part


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def get_flops(model):
    # 1. define a concrete function with a FIXED input shape
    input_signature = [
        tf.TensorSpec([1, 28, 28, 1], tf.float32, name='input_image')
    ]
    
    # 2. get the concrete function
    full_model = tf.function(lambda x: model(x))
    concrete_func = full_model.get_concrete_function(input_signature)

    # 3. freeze the graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    # 4. profile
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph,
        run_meta=run_meta, 
        cmd='op', 
        options=opts
    )
    
    # return 0 if the profiler fails (avoids crashing)
    return flops.total_float_ops if flops else 0

def benchmark_keras_model(model, test_images, num_samples=500):
    print(f"Benchmarking FP32 Baseline (Keras) on {num_samples} images...")
    
    latencies = []
    
    # loop over individual images to simulate "Batch Size = 1" (Real-time inference)
    for i in range(num_samples):
        img = test_images[i:i+1] # Shape (1, 28, 28, 1)
        
        start = time.time()
        # predict_on_batch is faster/lower overhead than .predict() for single images
        model.predict_on_batch(img) 
        end = time.time()
        
        # skip the first 10 runs (warm-up)
        if i >= 10:
            latencies.append((end - start) * 1000) # Convert to ms

    avg_latency = np.mean(latencies)
    return avg_latency
    

# 1. load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalize to [0, 1] for training (standard practice)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# add channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# defining model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    
    # block 1
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # block 2
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # block 3
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    
    # classifier
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4), # regularization to control overfitting [cite: 54]
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# 3. train the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train for a few epochs - usually converges quickly
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=64)

# 4. evaluation
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

# plotting curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')
plt.show()

# confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()

# simple profiler to estimate FLOPs for a single inference
flops = get_flops(model)
print(f"Total FLOPs: {flops:,}")
print(f"Est. MFLOPs per image: {flops / 1e6:.4f} M")

fp32_latency = benchmark_keras_model(model, x_test)
print(f"\n--- FP32 Baseline Results ---")
print(f"Latency: {fp32_latency:.4f} ms/img")
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
