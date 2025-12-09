import tensorflow as tf
import os
import cv2
import numpy as np

class WiderFaceLoader:
    def __init__(self, data_dir, annotation_file, img_height=112, img_width=112, batch_size=32):
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
        # Load file paths and boxes into memory
        self.img_paths, self.bboxes = self._parse_txt_annotations()
        print(f"Loaded {len(self.img_paths)} images from {self.annotation_file}")

    def _parse_txt_annotations(self):
        img_paths = []
        bboxes = []
        
        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            file_name = lines[i].strip()
            i += 1
            if i >= len(lines): break
            
            try:
                num_boxes = int(lines[i].strip())
            except ValueError:
                # Handle cases where file might be malformed
                i += 1
                continue
                
            i += 1
            
            current_faces = []
            for _ in range(num_boxes):
                if i >= len(lines): break
                # WIDER FACE format: x1 y1 w h blur expression ...
                coords = list(map(int, lines[i].strip().split()))
                x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                
                # Keep valid faces
                if w > 0 and h > 0:
                    current_faces.append((x, y, w, h))
                i += 1
            
            # Only add image if it has at least one face
            if len(current_faces) > 0:
                # Pick the largest face (Area = w * h)
                largest_face = max(current_faces, key=lambda b: b[2] * b[3])
                
                # Construct full path
                full_path = os.path.join(self.data_dir, file_name)
                img_paths.append(full_path)
                bboxes.append(largest_face)
                
        return img_paths, bboxes

    def _load_image_and_box(self, img_path, bbox):
        # 1. Read Image
        img_raw = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32) # Normalize [0,1]

        # 2. Get original dimensions
        original_shape = tf.shape(img)
        h_orig = tf.cast(original_shape[0], tf.float32)
        w_orig = tf.cast(original_shape[1], tf.float32)

        # 3. Resize Image
        img = tf.image.resize(img, [self.img_height, self.img_width])
        
        # 4. Normalize Bounding Box [x, y, w, h] -> [0.0 - 1.0]
        x = tf.cast(bbox[0], tf.float32) / w_orig
        y = tf.cast(bbox[1], tf.float32) / h_orig
        w = tf.cast(bbox[2], tf.float32) / w_orig
        h = tf.cast(bbox[3], tf.float32) / h_orig
        
        return img, tf.stack([x, y, w, h])

    def get_dataset(self):
        # Create dataset from memory lists
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.bboxes))
        
        # Shuffle and process
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.map(self._load_image_and_box, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batching
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset