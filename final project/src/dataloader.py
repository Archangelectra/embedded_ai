import tensorflow as tf
import os
import glob
import random
from typing import Tuple, Optional, List

class DigiFaceLoader:
    """
    A scalable TensorFlow data pipeline for the DigiFace-1M dataset.
    
    Attributes:
        data_dir (str): Path to the root directory containing identity subfolders.
        img_height (int): Target height for resizing.
        img_width (int): Target width for resizing.
        batch_size (int): Batch size for training.
    """

    def __init__(self, data_dir: str, img_height: int = 112, img_width: int = 112, batch_size: int = 32):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.autotune = tf.data.AUTOTUNE

    def _parse_image_and_label(self, file_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Reads an image file and extracts its label from the file path.
        
        Args:
            file_path: Tensor containing the string path to the image.
            
        Returns:
            Tuple of (processed_image, label_id).
        """
        # 1. Read the file
        img_raw = tf.io.read_file(file_path)
        
        # 2. Decode PNG
        # DigiFace images are RGBA (4 channels). We convert to RGB (3 channels).
        img = tf.io.decode_png(img_raw, channels=3)
        
        # 3. Convert to float32 and normalize to [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        # 4. Resize
        img = tf.image.resize(img, [self.img_height, self.img_width])
        
        # 5. Extract Label
        # Structure is: .../IdentityID/ImageID.png
        # We take the parent folder name as the label.
        parts = tf.strings.split(file_path, os.sep)
        # The identity ID is the second to last element
        label_str = parts[-2]
        
        # Convert string label (e.g., "10234") to integer
        # Note: If your folders have non-numeric prefixes, you might need a hash table here.
        # DigiFace usually uses numeric folder names.
        label = tf.strings.to_number(label_str, out_type=tf.int32)
        
        return img, label

    def _augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies random augmentations suitable for face recognition.
        """
        # Random horizontal flip (mirroring)
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness/contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random saturation
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        
        return image, label

    def get_dataset(self, validation_split: float = 0.2, subset: str = 'training', seed: int = 42, shuffle_buffer: int = 10000) -> tf.data.Dataset:
        """
        Creates and returns a configured tf.data.Dataset with proper Identity-level splitting.
        
        Args:
            validation_split (float): Percentage of *identities* to use for validation.
            subset (str): 'training' or 'validation'.
            seed (int): Random seed for reproducible splitting.
            shuffle_buffer (int): Buffer size for shuffling images.
            
        Returns:
            A prefetch-optimized tf.data.Dataset yielding (batch_images, batch_labels).
        """
        # 1. Gather all identity folder paths
        # We scan the directory to find all subfolders (each representing an identity)
        print(f"Scanning directories in {self.data_dir}...")
        all_folders = glob.glob(os.path.join(self.data_dir, "*"))
        all_folders = [f for f in all_folders if os.path.isdir(f)]
        
        # 2. Sort and Shuffle Deterministically
        # Sorting ensures the starting order is always the same regardless of OS file system
        all_folders.sort()
        # Shuffling ensures we get a random distribution of identities
        random.Random(seed).shuffle(all_folders)
        
        total_identities = len(all_folders)
        if total_identities == 0:
            raise ValueError(f"No identity folders found in {self.data_dir}")

        # 3. Perform the Split
        split_idx = int(total_identities * (1 - validation_split))
        
        if subset == 'training':
            selected_folders = all_folders[:split_idx]
        elif subset == 'validation':
            selected_folders = all_folders[split_idx:]
        else:
            raise ValueError("Subset must be 'training' or 'validation'")
            
        print(f"[{subset.upper()}] Selected {len(selected_folders)} identities out of {total_identities} total.")

        # 4. Create Dataset from Folder Paths
        # This is more efficient than listing 1M files at once.
        ds = tf.data.Dataset.from_tensor_slices(selected_folders)

        # Shuffle the order of identities (only for training)
        if subset == 'training':
            ds = ds.shuffle(buffer_size=len(selected_folders))

        # 5. Interleave: Process folders in parallel to yield images
        # This function takes a folder path and lists all .pngs inside it
        def _list_images_in_folder(folder_path):
            return tf.data.Dataset.list_files(tf.strings.join([folder_path, os.sep, "*.png"]), shuffle=True)

        # Cycle_length controls how many folders we process concurrently.
        # This mixes images from different identities in the batch.
        ds = ds.interleave(
            _list_images_in_folder,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=(subset != 'training') # Deterministic order for validation
        )

        # 6. Parse Images (Load, decode, resize)
        ds = ds.map(self._parse_image_and_label, num_parallel_calls=self.autotune)

        # 7. Augment (Training only)
        if subset == 'training':
            ds = ds.map(self._augment, num_parallel_calls=self.autotune)
            # Shuffle individual images within the stream
            ds = ds.shuffle(buffer_size=shuffle_buffer)

        # 8. Batch and Prefetch
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds