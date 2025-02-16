import os
import cv2
import numpy as np
import tensorflow as tf

# Paths
model_path = r"C:\Users\Aditya\Downloads\testgaip\unet_model.h5"  # Path to saved model
unseen_images_dir = r"C:\Users\Aditya\Downloads\testgaip\unseen_images"  # Directory of unseen images
output_mask_dir = r"C:\Users\Aditya\Downloads\testgaip\predicted_masks"  # Directory to save predicted masks
img_size = (256, 256)  # Input size for the model

# Function to predict masks on unseen images
def predict_masks_on_unseen_images(image_dir, model_path, img_size=(256, 256)):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    print("Segmentation model loaded successfully.")
    
    # Get all image paths
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg"))]
    
    # Predict masks for each image
    masks = []
    for image_path in image_paths:
        # Load and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, img_size) / 255.0  # Normalize to [0, 1]
        resized_image = np.expand_dims(resized_image, axis=(0, -1))  # Add batch and channel dimensions

        # Predict the mask
        predicted_mask = model.predict(resized_image)[0]  # Remove batch dimension
        predicted_classes = np.argmax(predicted_mask, axis=-1)  # Convert to class indices
        masks.append(predicted_classes)
    
    return masks, image_paths

# Function to save predicted masks
def save_masks(masks, image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        # Save mask as a grayscale image
        output_path = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(output_path, (mask * 127).astype(np.uint8))  # Scale mask values for visualization
        print(f"Saved mask to {output_path}")

# Main script for predictions
if __name__ == "__main__":
    # Predict masks for unseen images
    masks, image_paths = predict_masks_on_unseen_images(unseen_images_dir, model_path, img_size)

    # Save the predicted masks
    save_masks(masks, image_paths, output_mask_dir)

    # Optional: Display a few predictions
    import matplotlib.pyplot as plt
    for i in range(min(3, len(image_paths))):  # Display up to 3 images
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = masks[i]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="viridis")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.show()
