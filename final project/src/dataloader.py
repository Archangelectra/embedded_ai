import tensorflow as tf
import tensorflow_datasets as tfds

class WiderFaceLoader:
    def __init__(self, img_height: int = 112, img_width: int = 112, batch_size: int = 32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def _preprocess(self, sample):
        # 1. Get Image and Faces
        image = sample['image']
        # TFDS bboxes are [ymin, xmin, ymax, xmax] (normalized 0-1)
        bboxes = sample['faces']['bbox'] 
        
        # 2. Find the Largest Face (by area)
        # We calculate height * width for all boxes
        # height = (ymax - ymin), width = (xmax - xmin)
        heights = bboxes[:, 2] - bboxes[:, 0]
        widths = bboxes[:, 3] - bboxes[:, 1]
        areas = heights * widths
        
        # Find index of largest face
        largest_idx = tf.argmax(areas)
        best_box = bboxes[largest_idx]
        
        # 3. Convert Format: [ymin, xmin, ymax, xmax] -> [x, y, w, h]
        # Your model expects x (left), y (top), w, h
        ymin, xmin, ymax, xmax = best_box[0], best_box[1], best_box[2], best_box[3]
        
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        
        # 4. Resize Image
        image = tf.image.convert_image_dtype(image, tf.float32) # [0,1]
        image = tf.image.resize(image, [self.img_height, self.img_width])
        
        return image, tf.stack([x, y, w, h])

    def _filter_no_faces(self, sample):
        # Filter out images that have 0 faces
        return tf.shape(sample['faces']['bbox'])[0] > 0

    def get_dataset(self, split='train'):
        # Load WIDER FACE from tensorflow_datasets
        # Note: This will download 3GB+ on the first run.
        ds = tfds.load('wider_face', split=split, shuffle_files=True)
        
        # Filter images with no faces
        ds = ds.filter(self._filter_no_faces)
        
        # Map to our format (Image, [x,y,w,h])
        ds = ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and Prefetch
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds