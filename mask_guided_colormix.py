
"""
Mask-Guided Colour-Mix Data Augmentation
--------------------------------------
This script implements the Mask-Guided Colour-Mix augmentation technique
for plant disease severity estimation, preserving leaf anatomy while
blending background information from same-severity samples.

Author: Sonu Varghese K
License: MIT
"""

import cv2
import numpy as np
import os
import random
import shutil


def get_hsv_mask(image_path, lower_hsv, upper_hsv):
    """
    Load an image, convert it to HSV color space, and generate a binary mask.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_hsv, upper_hsv)


def extract_area_with_mask(image_path, lower_hsv, upper_hsv):
    """
    Extract a specific region from an image using an HSV mask.
    Returns the RGB image, extracted region, and mask.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    mask = get_hsv_mask(image_path, lower_hsv, upper_hsv)
    if mask is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    return img_rgb, extracted, mask


def process_and_combine_images(img1_path, img2_path,
                               lower_leaf_green, upper_leaf_green,
                               lower_lesion_brownish, upper_lesion_brownish,
                               lower_lesion_yellowish, upper_lesion_yellowish,
                               lower_lesion_dark, upper_lesion_dark):
    """
    Generate a Colour-Mix composite image by preserving
    leaf + lesion regions from image 1 and background from image 2.
    """
    img1_rgb, _, leaf_mask = extract_area_with_mask(
        img1_path, lower_leaf_green, upper_leaf_green
    )
    img2 = cv2.imread(img2_path)

    if img1_rgb is None or img2 is None or leaf_mask is None:
        return None

    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if img1_rgb.shape != img2_rgb.shape:
        img2_rgb = cv2.resize(
            img2_rgb,
            (img1_rgb.shape[1], img1_rgb.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    _, _, mask_brown = extract_area_with_mask(
        img1_path, lower_lesion_brownish, upper_lesion_brownish
    )
    _, _, mask_yellow = extract_area_with_mask(
        img1_path, lower_lesion_yellowish, upper_lesion_yellowish
    )
    _, _, mask_dark = extract_area_with_mask(
        img1_path, lower_lesion_dark, upper_lesion_dark
    )

    lesion_mask = np.zeros(img1_rgb.shape[:2], dtype=np.uint8)
    for m in [mask_brown, mask_yellow, mask_dark]:
        if m is not None:
            lesion_mask = cv2.bitwise_or(lesion_mask, m)

    lesion_on_leaf = cv2.bitwise_and(lesion_mask, leaf_mask)
    preserve_mask = cv2.bitwise_or(leaf_mask, lesion_on_leaf)

    preserved = cv2.bitwise_and(img1_rgb, img1_rgb, mask=preserve_mask)
    inverse_mask = cv2.bitwise_not(preserve_mask)
    background = cv2.bitwise_and(img2_rgb, img2_rgb, mask=inverse_mask)

    return cv2.add(preserved, background)


def main():
    lower_leaf_green = np.array([30, 30, 20])
    upper_leaf_green = np.array([80, 250, 250])

    lower_lesion_brownish = np.array([0, 50, 20])
    upper_lesion_brownish = np.array([20, 255, 150])

    lower_lesion_yellowish = np.array([20, 50, 50])
    upper_lesion_yellowish = np.array([40, 255, 255])

    lower_lesion_dark = np.array([0, 0, 0])
    upper_lesion_dark = np.array([180, 255, 70])

    base_dir = "data/original_dataset"
    output_dir = "data/colourmix_dataset"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    severity_levels = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for level in severity_levels:
        src = os.path.join(base_dir, level)
        dst = os.path.join(output_dir, level)
        os.makedirs(dst, exist_ok=True)

        images = [
            os.path.join(src, f)
            for f in os.listdir(src)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        for img1 in images:
            img2 = random.choice(images)
            if img1 == img2:
                continue

            composite = process_and_combine_images(
                img1, img2,
                lower_leaf_green, upper_leaf_green,
                lower_lesion_brownish, upper_lesion_brownish,
                lower_lesion_yellowish, upper_lesion_yellowish,
                lower_lesion_dark, upper_lesion_dark
            )

            if composite is not None:
                name = os.path.splitext(os.path.basename(img1))[0]
                out_path = os.path.join(dst, f"{name}_colourmix.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
