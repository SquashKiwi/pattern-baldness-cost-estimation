import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import albumentations as A
import tensorflow as tf

# Paths to your directories
mask_dir = r"C:\Users\Aditya\Downloads\testgaip\masks"
image_dir = r"C:\Users\Aditya\Downloads\testgaip\ogimages"
output_mask_dir = r"C:\Users\Aditya\Downloads\testgaip\predicted_masks"  # For saving predicted masks

# Function to parse filenames for matching images and masks
def parse_filename(filename, is_mask=True):
    filename = os.path.splitext(filename)[0]
    if is_mask:
        parts = filename.split('_')
        group = int(parts[0].replace('group', ''))
        number = int(parts[1])
    else:
        parts = filename.split('_')
        group = int(parts[0].replace('Class', ''))
        number = int(parts[1])
    return group, number

# Function to map grayscale mask values to segmentation classes
def map_grayscale_to_classes(mask):
    """
    Map grayscale values to specific classes:
    - 0: Background
    - 1: Hair
    - 2: Scalp
    """
    encoded_mask = np.zeros_like(mask, dtype=np.uint8)
    encoded_mask[(mask >= 50) & (mask <= 100)] = 1  # Hair
    encoded_mask[(mask > 100) & (mask <= 130)] = 2  # Scalp
    return encoded_mask

# Define augmentation pipeline
def get_augmentation_pipeline(img_size=(256, 256)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.Resize(img_size[0], img_size[1])  # Ensure consistent size
    ])

# Function to prepare images and masks with augmentation
def prepare_data_with_augmentation(image_dir, mask_dir, img_size=(256, 256), augment=True, n_augmentations=5):
    images = []
    masks = []
    unmatched_files = []
    
    # Load filenames
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)

    # Define augmentation pipeline
    augmentation_pipeline = get_augmentation_pipeline(img_size)

    for mask_file in mask_files:
        mask_group, mask_number = parse_filename(mask_file, is_mask=True)

        # Find the corresponding image
        matching_image_file = None
        for image_file in image_files:
            image_group, image_number = parse_filename(image_file, is_mask=False)
            if image_group == mask_group and image_number == mask_number:
                matching_image_file = image_file
                break

        if matching_image_file:
            img_path = os.path.join(image_dir, matching_image_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # Load image and mask
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Resize and preprocess
            image = cv2.resize(image, img_size) / 255.0  # Normalize
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            encoded_mask = map_grayscale_to_classes(mask)

            # Augment data
            if augment:
                for _ in range(n_augmentations):
                    augmented = get_augmentation_pipeline(img_size)(image=image, mask=encoded_mask)
                    images.append(augmented['image'])
                    masks.append(augmented['mask'])

            # Include original data
            images.append(image)
            masks.append(encoded_mask)
        else:
            unmatched_files.append(mask_file)

    if unmatched_files:
        print(f"Unmatched files: {unmatched_files}")

    # Convert to NumPy arrays
    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)
    return images, masks

# Define the U-Net model
def unet_model(input_size=(256, 256, 1), num_classes=3):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    return models.Model(inputs, outputs)

# Function to predict masks on unseen images
def predict_masks_on_unseen_images(image_dir, model_path, img_size=(256, 256)):
    model = tf.keras.models.load_model(model_path)
    print("Segmentation model loaded successfully.")
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg"))]
    masks = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, img_size) / 255.0
        resized_image = np.expand_dims(resized_image, axis=(0, -1))
        predicted_mask = model.predict(resized_image)[0]
        masks.append(np.argmax(predicted_mask, axis=-1))
    return masks, image_paths

# Save predicted masks
def save_masks(masks, image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        output_path = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(output_path, (mask * 127).astype(np.uint8))  # Scale for visualization
        print(f"Saved mask to {output_path}")

# Main script
if __name__ == "__main__":
    # Prepare data and train the model
    img_size = (256, 256)
    images, masks = prepare_data_with_augmentation(image_dir, mask_dir, img_size, augment=True, n_augmentations=10)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    model = unet_model(input_size=(256, 256, 1), num_classes=3)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=20)

    # Save the trained model
    model_path = r"C:\Users\Aditya\Downloads\testgaip\unet_model.h5"
    model.save(model_path)
    print(f"Segmentation model saved to {model_path}")

    # Predict and save masks for unseen images
    unseen_images_dir = r"C:\Users\Aditya\Downloads\testgaip\unseen_images"
    masks, image_paths = predict_masks_on_unseen_images(unseen_images_dir, model_path, img_size)
    save_masks(masks, image_paths, output_mask_dir)
