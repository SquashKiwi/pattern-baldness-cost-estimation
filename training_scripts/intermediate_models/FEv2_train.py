import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Dataset Directory
DATA_DIR = "./augmented"

# Load Images and Labels
def load_dataset(data_dir, image_size=(256, 256)):
    images = []
    masks = []
    label_map = {"group1": 1, "group2": 0}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            group = filename.split("_")[0]
            if group in label_map:
                filepath = os.path.join(data_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, image_size) / 255.0  # Normalize
                
                # Generate binary mask
                mask = np.full(image_size, label_map[group], dtype=np.float32)
                mask = mask.reshape(image_size + (1,))  # Add channel dimension
                
                images.append(img)
                masks.append(mask)
    
    images = np.array(images).reshape(-1, image_size[0], image_size[1], 1)
    masks = np.array(masks)
    return images, masks

# Define U-Net Model
def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Contracting Path
    c1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=0.1)(c1)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c1)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=0.1)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(p1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=0.1)(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c2)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=0.1)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(p2)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=0.1)(c3)
    c3 = Dropout(0.3)(c3)
    c3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(c3)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=0.1)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(p3)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU(alpha=0.1)(c4)
    c4 = Dropout(0.4)(c4)
    c4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(c4)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU(alpha=0.1)(c4)

    # Expansive Path
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(u5)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU(alpha=0.1)(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c5)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU(alpha=0.1)(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(u6)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU(alpha=0.1)(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c6)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU(alpha=0.1)(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(u7)
    c7 = BatchNormalization()(c7)
    c7 = LeakyReLU(alpha=0.1)(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c7)
    c7 = BatchNormalization()(c7)
    c7 = LeakyReLU(alpha=0.1)(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load Data
X, Y = load_dataset(DATA_DIR)

# Split Data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Callbacks
model_checkpoint = ModelCheckpoint('fe2rev.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
model = unet_model()
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=20,
    callbacks=[model_checkpoint, reduce_lr, early_stopping]
)

# Save Model
model.save('fe2.keras', save_format='keras')
