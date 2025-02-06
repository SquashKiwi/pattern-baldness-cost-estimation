import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import math

# Dataset Directory
DATA_DIR = "./augmented"

# Load Images and Labels
def load_dataset(data_dir, image_size=(256, 256)):
    images = []
    labels = []
    label_map = {"group1": 0, "group2": 1, "group3": 2, "group4": 3}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            group = filename.split("_")[0]
            if group in label_map:
                filepath = os.path.join(data_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, image_size)
                img = img / 255.0
                images.append(img)
                labels.append(label_map[group])
    
    images = np.array(images).reshape(-1, image_size[0], image_size[1], 1)
    labels = np.array(labels)
    return images, labels

# Define a Simpler CNN for Classification
def simple_cnn(input_size=(256, 256, 1), num_classes=4):
    inputs = Input(input_size)

    # Layer 1
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Layer 2
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Layer 3
    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Learning Rate Scheduler
def cosine_annealing(epoch, lr, total_epochs=20, min_lr=1e-5, max_lr=1e-3):
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

# Load Data
X, Y = load_dataset(DATA_DIR)

# Split Data
# Step 1: Split into training + validation and test
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Split training + validation into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)
# Note: 0.25 of 80% = 20%, so the final split is 60% train, 20% validation, 20% test

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Testing set: {len(X_test)} samples")

# Callbacks
model_checkpoint = ModelCheckpoint("simple_cnn_model.keras", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lambda epoch, lr: cosine_annealing(epoch, lr))

# Train Model
model = simple_cnn()
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,  # Reduced batch size for stability
    epochs=20,
    callbacks=[model_checkpoint, early_stopping, lr_scheduler]
)

# Save Model
model.save("fe4.keras", save_format="keras")

# Evaluate on Test Set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
