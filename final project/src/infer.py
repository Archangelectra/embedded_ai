import cv2
import numpy as np

def infer(model, frame):
    """
    inference method
    """
    h_img, w_img, _ = frame.shape
    
    # 1. preprocess (match training size)
    input_size = (112, 112)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    
    # normalize to [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # batch dimension
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # 2. predict
    # output shape will be (1, 4) -> [[x, y, w, h]] normalized
    prediction = model.predict(input_tensor, verbose=0)
    box = prediction[0] # Get the single box
    
    # 3. convert normalized coordinates back to pixels
    # prediction is [x_norm, y_norm, w_norm, h_norm]
    pred_x, pred_y, pred_w, pred_h = box
    
    x = int(pred_x * w_img)
    y = int(pred_y * h_img)
    w = int(pred_w * w_img)
    h = int(pred_h * h_img)
    
    # return as list of tuples (to keep runner.py happy)
    return [(x, y, w, h)]