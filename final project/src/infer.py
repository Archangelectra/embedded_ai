import cv2
import numpy as np
import tensorflow as tf

def infer(model, frame):
    """
    Detects faces in a frame using a Keras detection model.
    
    Args:
        model: A loaded Keras model (tf.keras.Model).
        frame: The video frame (BGR).
        
    Returns:
        List of tuples (x, y, w, h) for detected faces.
    """
    h_img, w_img, _ = frame.shape
    
    # 1. Preprocess the image
    # Note: Adjust input_shape to match your specific detection model (e.g., 300x300, 320x320)
    input_shape = (300, 300) 
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, input_shape)
    
    # Expand dims (1, H, W, C)
    input_tensor = np.expand_dims(img_resized, axis=0)
    
    # Normalize (Check your model's requirements: [0,1] or [-1,1])
    # Common for MobileNet-based models:
    input_tensor = (input_tensor.astype(np.float32) - 127.5) / 127.5

    # 2. Run Inference
    # Detection models usually return a list of tensors: [boxes, classes, scores, num_detections]
    detections = model.predict(input_tensor, verbose=0)
    
    # 3. Parse Output
    # This structure depends heavily on the specific model architecture.
    # Assuming a standard SSD output where:
    # detections[0] = boxes (batch, num_boxes, 4)
    # detections[1] = scores (batch, num_boxes)
    # detections[2] = classes (batch, num_boxes)
    
    # Note: Keras models might return a single dictionary or list. 
    # If detections is a list:
    boxes = detections[0][0] # Remove batch dim
    scores = detections[1][0]
    
    results = []
    confidence_threshold = 0.5

    for i, score in enumerate(scores):
        if score > confidence_threshold:
            # Box format: y_min, x_min, y_max, x_max (normalized)
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Convert to pixels
            x = int(xmin * w_img)
            y = int(ymin * h_img)
            w = int((xmax - xmin) * w_img)
            h = int((ymax - ymin) * h_img)
            
            if w > 0 and h > 0:
                results.append((x, y, w, h))
                
    return results