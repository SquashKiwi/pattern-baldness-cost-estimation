import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Dataset Directory
DATA_DIR = "./augmented"
CATEGORIES = ["group1", "group2", "group3", "group4"]
IMG_SIZE = (256, 256)

# Load Dataset
def load_dataset(data_dir, img_size=(256, 256)):
    images = []
    labels = []
    label_map = {category: idx for idx, category in enumerate(CATEGORIES)}
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            group = filename.split("_")[0]
            if group in label_map:
                filepath = os.path.join(data_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size) / 255.0
                images.append(img)
                labels.append(label_map[group])
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels)
    return images, labels

# Enhanced SqueezeNet Model with Regularization
def enhanced_squeezenet(input_shape=(256, 256, 1), num_classes=4):
    def fire_module(x, squeeze, expand):
        x = Conv2D(squeeze, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        expand_1x1 = Conv2D(expand, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        expand_1x1 = BatchNormalization()(expand_1x1)
        expand_1x1 = Activation("relu")(expand_1x1)
        expand_3x3 = Conv2D(expand, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        expand_3x3 = BatchNormalization()(expand_3x3)
        expand_3x3 = Activation("relu")(expand_3x3)
        x = tf.keras.layers.concatenate([expand_1x1, expand_3x3])
        return x

    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(96, (7, 7), strides=(2, 2), padding="valid", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Fire Modules
    x = fire_module(x, squeeze=16, expand=64)
    x = fire_module(x, squeeze=16, expand=64)
    x = fire_module(x, squeeze=32, expand=128)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze=32, expand=128)
    x = fire_module(x, squeeze=48, expand=192)
    x = fire_module(x, squeeze=48, expand=192)
    x = fire_module(x, squeeze=64, expand=256)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze=64, expand=256)

    # Dropout Regularization
    x = Dropout(0.6)(x)

    # Final Conv Layer
    x = Conv2D(num_classes, (1, 1), activation="relu", padding="valid", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = GlobalAveragePooling2D()(x)

    # Classification Layer
    outputs = Activation("softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Load Data
X, Y = load_dataset(DATA_DIR)

# Train-Validation-Test Split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Callbacks
model_checkpoint = ModelCheckpoint("enhanced_squeezenet_model.keras", monitor="val_loss", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train Model
model = enhanced_squeezenet(input_shape=(256, 256, 1), num_classes=len(CATEGORIES))
model.summary()
model.fit(
    datagen.flow(X_train, Y_train, batch_size=16),
    validation_data=(X_val, Y_val),
    steps_per_epoch=len(X_train) // 16,
    epochs=20,
    callbacks=[model_checkpoint, early_stopping, reduce_lr]
)

# Evaluate on Test Set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
