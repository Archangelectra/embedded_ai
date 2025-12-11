import tensorflow as tf
import cv2
import numpy as np
import os
from dataloader import WiderFaceLoader

# 1. setup loader (same settings as train.py)
IMG_SIZE = 112
loader = WiderFaceLoader(
    data_dir="../data/WIDER_train/images",
    annotation_file="../data/wider_face_split/wider_face_train_bbx_gt.txt",
    img_height=IMG_SIZE,
    img_width=IMG_SIZE,
    batch_size=8 
)

# 2. get one batch
dataset = loader.get_dataset()
for images, boxes in dataset.take(1):
    # images shape: (8, 112, 112, 3), range [-1, 1]
    # boxes shape: (8, 4), range [0, 1]
    
    print("--- DEBUG BATCH INFO ---")
    print(f"Image Min: {np.min(images):.2f}, Max: {np.max(images):.2f}")
    print(f"Box Example (Normalized): {boxes[0].numpy()}")

    # 3. visualization
    canvas = []
    
    for i in range(len(images)):
        img = images[i].numpy()
        box = boxes[i].numpy()
        
        # denormalize Image: [-1, 1] -> [0, 255]
        # (img + 1) / 2 * 255
        img_vis = ((img + 1.0) / 2.0 * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        # denormalize Box: [0, 1] -> [0, 112]
        x = int(box[0] * IMG_SIZE)
        y = int(box[1] * IMG_SIZE)
        w = int(box[2] * IMG_SIZE)
        h = int(box[3] * IMG_SIZE)
        
        # draw
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_vis, f"{w}x{h}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        canvas.append(img_vis)

    # Save the debug image
    final_grid = np.hstack(canvas) # Stitch images side-by-side
    cv2.imwrite("debug_data_check.jpg", final_grid)
    print("Saved 'debug_data_check.jpg'. Check this image!")