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
        
        # load file paths and boxes into memory
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
                i += 1
                continue
                
            i += 1
            
            current_faces = []
            for _ in range(num_boxes):
                if i >= len(lines): break
                coords = list(map(int, lines[i].strip().split()))
                x, y, w, h = coords[0], coords[1], coords[2], coords[3]
                
                if w > 0 and h > 0:
                    current_faces.append((x, y, w, h))
                i += 1
            
            if len(current_faces) > 0:
                # pick the largest face (area = w * h)
                largest_face = max(current_faces, key=lambda b: b[2] * b[3])
                
                full_path = os.path.join(self.data_dir, file_name)
                img_paths.append(full_path)
                bboxes.append(largest_face)
                
        return img_paths, bboxes

    def _load_image_and_box(self, img_path, bbox):
        # 1. read image
        img_raw = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32) 

        # normalize [-1, 1]
        img = (img * 2.0) - 1.0

        # 2. get original dimensions
        original_shape = tf.shape(img)
        h_orig = tf.cast(original_shape[0], tf.float32)
        w_orig = tf.cast(original_shape[1], tf.float32)

        # 3. resize image
        img = tf.image.resize(img, [self.img_height, self.img_width])
        
        # 4. normalize bounding box [x, y, w, h] -> [0.0 - 1.0]
        x = tf.cast(bbox[0], tf.float32) / w_orig
        y = tf.cast(bbox[1], tf.float32) / h_orig
        w = tf.cast(bbox[2], tf.float32) / w_orig
        h = tf.cast(bbox[3], tf.float32) / h_orig
        
        final_box = tf.stack([x, y, w, h])
        
        if tf.random.uniform(()) > 0.5:
            # flip Image
            img = tf.image.flip_left_right(img)
            
            # flip box
            # new X = 1.0 - (old_X + Width)
            new_x = 1.0 - (x + w)
            final_box = tf.stack([new_x, y, w, h])
            
        return img, final_box

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.bboxes))
        
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.map(self._load_image_and_box, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset