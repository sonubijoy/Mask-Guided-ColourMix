
"""
Model Training and Evaluation for Rice Sheath Blight Severity Classification
---------------------------------------------------------------------------
This script trains and evaluates CNN-based models for multi-class plant disease
severity classification. It supports custom CNN architectures and can be
extended to transfer-learning models (VGG16, ResNet50, InceptionV3, etc.).

Author: Sonu Varghese K
License: MIT
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


# ---------------- Configuration ----------------
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 35


# ---------------- Data Generators ----------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)


# ---------------- Model Definitions ----------------
def create_improved_cnn_model(input_shape, num_classes, l2_reg=0.001):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               kernel_regularizer=l2(l2_reg),
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu',
              kernel_regularizer=l2(l2_reg)),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------- Training & Evaluation ----------------
def train_and_evaluate(model, model_name, epochs):
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=1
    )

    training_time = time.time() - start_time

    print(f"\n--- Evaluating {model_name} ---")
    loss, accuracy = model.evaluate(test_generator, verbose=0)

    y_pred = model.predict(test_generator)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred_labels)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.show()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "training_time": training_time
    }


# ---------------- Main ----------------
if __name__ == "__main__":
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = create_improved_cnn_model(input_shape, NUM_CLASSES)
    results = train_and_evaluate(
        model,
        "Improved Custom CNN",
        epochs=EPOCHS
    )

    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
