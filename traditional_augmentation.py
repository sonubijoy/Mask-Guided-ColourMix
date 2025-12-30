
"""
Traditional Image Augmentation with Class Balancing
--------------------------------------------------
This script performs traditional data augmentation (flip, rotation,
color jitter, and biased random cropping) to balance class distributions
in plant disease severity datasets.

Author: Sonu Varghese K
License: MIT
"""

import tensorflow as tf
import os
import shutil
from PIL import Image
import math


def augment_data(image, label):
    """
    Apply traditional augmentations to a single image.
    Image is expected to be in float range [0.0, 1.0].
    """
    image = tf.image.rot90(image, k=3)
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.08)

    h, w = tf.shape(image)[0], tf.shape(image)[1]
    crop_scale = tf.random.uniform([], 0.6, 0.9)

    crop_h = tf.cast(tf.cast(h, tf.float32) * crop_scale, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * crop_scale, tf.int32)

    max_off_h = h - crop_h
    max_off_w = w - crop_w

    min_bias_h = tf.cast(tf.cast(max_off_h, tf.float32) * 0.4, tf.int32)
    off_h = tf.random.uniform([], min_bias_h, max_off_h + 1, dtype=tf.int32)
    off_w = tf.random.uniform([], 0, max_off_w + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, off_h, off_w, crop_h, crop_w)
    image = tf.image.resize(image, (h, w))

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def load_and_preprocess_image(image_path, label):
    """Load and preprocess image from disk."""
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    except tf.errors.InvalidArgumentError:
        return None, label


def augment_and_save_images_balanced(data_dir, output_dir):
    """
    Perform class-balanced traditional augmentation and save results.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    class_names = sorted(os.listdir(data_dir))
    class_counts = {}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            imgs = [f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[cls] = len(imgs)
        else:
            class_counts[cls] = 0

    target_count = max(class_counts.values())
    if target_count == 0:
        print("No images found.")
        return

    print("Initial class counts:", class_counts)
    print("Target count per class:", target_count)

    for cls in class_names:
        src = os.path.join(data_dir, cls)
        dst = os.path.join(output_dir, cls)
        os.makedirs(dst, exist_ok=True)

        imgs = [f for f in os.listdir(src)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img in imgs:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

        needed = target_count - len(imgs)
        if needed <= 0:
            continue

        factor = math.ceil(needed / len(imgs))
        label = class_names.index(cls)

        for img in imgs:
            img_path = os.path.join(src, img)
            image_tensor, _ = load_and_preprocess_image(img_path, label)
            if image_tensor is None:
                continue

            for i in range(factor):
                aug_img, _ = augment_data(image_tensor, label)
                aug_img = tf.image.convert_image_dtype(aug_img, tf.uint8).numpy()
                save_path = os.path.join(dst, f"aug_{i}_{img}")
                Image.fromarray(aug_img).save(save_path)


if __name__ == "__main__":
    DATA_DIR = "data/original_dataset"
    OUTPUT_DIR = "data/traditional_augmented_dataset"
    augment_and_save_images_balanced(DATA_DIR, OUTPUT_DIR)
