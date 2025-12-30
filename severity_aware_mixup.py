
"""
Severity-Aware Mixup Augmentation for Plant Disease Datasets
------------------------------------------------------------
This script implements a severity-aware Mixup strategy where images from
different disease severity levels are linearly interpolated using a fixed
lambda, and the resulting severity label is computed as the floor of the
average severity.

Author: Sonu Varghese K
License: MIT
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import random
import math
import shutil
from tqdm import tqdm

# ---------------- Configuration ----------------
DATASET_BASE_PATH = "data/original_dataset"   # Update to your dataset root
OUTPUT_MIXED_IMAGES_DIR = "data/mixup_dataset"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_SEVERITY_LEVELS = 6
ASSIGNED_LAMBDA_VALUE = 0.5


# ---------------- Helper Functions ----------------
def load_image(image_path):
    """Load image, convert to RGB, normalize, and resize."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return image


def simple_mixup_assigned_severity(img_a, severity_a, img_b, severity_b, assigned_lambda):
    """
    Apply Mixup with a fixed lambda and compute severity as floor of average.
    """
    lambda_val = tf.constant(assigned_lambda, dtype=tf.float32)
    mixed_img = lambda_val * img_a + (1 - lambda_val) * img_b
    mixed_severity = math.floor((severity_a + severity_b) / 2.0)
    return mixed_img, mixed_severity, lambda_val


# ---------------- Main Pipeline ----------------
def main():
    if os.path.exists(OUTPUT_MIXED_IMAGES_DIR):
        shutil.rmtree(OUTPUT_MIXED_IMAGES_DIR)
    os.makedirs(OUTPUT_MIXED_IMAGES_DIR, exist_ok=True)

    all_image_paths = {}
    for i in range(1, NUM_SEVERITY_LEVELS + 1):
        level_dir = os.path.join(DATASET_BASE_PATH, f"Stage-{i}")
        if os.path.isdir(level_dir):
            images = [
                os.path.join(level_dir, f)
                for f in os.listdir(level_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            all_image_paths[i] = images
        else:
            all_image_paths[i] = []

    mix_counter = 0

    for severity_a in range(1, NUM_SEVERITY_LEVELS + 1):
        for img_a_path in tqdm(all_image_paths[severity_a], desc=f"Mixing Level {severity_a}"):
            img_a = load_image(img_a_path)
            if img_a is None:
                continue

            for severity_b in range(1, NUM_SEVERITY_LEVELS + 1):
                if severity_a == severity_b and severity_a != 6:
                    continue

                img_b_path = random.choice(all_image_paths[severity_b])
                img_b = load_image(img_b_path)
                if img_b is None:
                    continue

                mixed_img, mixed_sev, _ = simple_mixup_assigned_severity(
                    img_a, severity_a, img_b, severity_b, ASSIGNED_LAMBDA_VALUE
                )

                mixed_np = (mixed_img.numpy() * 255).astype(np.uint8)
                target_dir = os.path.join(OUTPUT_MIXED_IMAGES_DIR, f"level{mixed_sev}")
                os.makedirs(target_dir, exist_ok=True)

                filename = f"mix_L{severity_a}_L{severity_b}_{random.randint(0,99999)}.jpg"
                save_path = os.path.join(target_dir, filename)
                cv2.imwrite(save_path, cv2.cvtColor(mixed_np, cv2.COLOR_RGB2BGR))
                mix_counter += 1

    print(f"Total mixed images generated: {mix_counter}")


if __name__ == "__main__":
    main()
