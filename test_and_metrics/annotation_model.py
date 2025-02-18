import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
segmentation_model_path = r"C:\Users\Aditya\Downloads\testgaip\unet_model.h5"  # Path to segmentation model
classification_model_path = r"C:\Users\Aditya\Downloads\testgaip\baldness_classifier.keras"  # Path to classification model
unseen_images_dir = r"C:\Users\Aditya\Downloads\testgaip\unseen_images"  # Directory of unseen images
output_mask_dir = r"C:\Users\Aditya\Downloads\testgaip\predicted_masks"  # Directory to save predicted masks
img_size = (256, 256)  # Input size for both models
categories = ["group1", "group2", "group3", "group4"]  # Classification categories

# Load the models
segmentation_model = tf.keras.models.load_model(segmentation_model_path)
classification_model = tf.keras.models.load_model(classification_model_path)

print("Models loaded successfully.")

# Function to predict masks on unseen images
def predict_mask(image_path, model, img_size):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, img_size) / 255.0  # Normalize to [0, 1]
    resized_image = np.expand_dims(resized_image, axis=(0, -1))  # Add batch and channel dimensions

    # Predict the mask
    predicted_mask = model.predict(resized_image)[0]  # Remove batch dimension
    predicted_classes = np.argmax(predicted_mask, axis=-1)  # Convert to class indices
    return predicted_classes

# Function to classify using the mask
def classify_image_with_mask(image_path, mask, classification_model, categories, img_size):
    # Apply the mask to the original image
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, img_size)

    # Create a masked image
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=(mask.astype(np.uint8)))

    # Preprocess the masked image for classification
    masked_image = masked_image / 255.0  # Normalize to [0, 1]
    masked_image = np.expand_dims(masked_image, axis=0)  # Add batch dimension

    # Predict classification
    prediction = classification_model.predict(masked_image)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Confidence percentage

    return predicted_class, confidence

# Function to save the predicted mask
def save_mask(mask, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, (mask * 127).astype(np.uint8))  # Scale mask values for visualization
    print(f"Saved mask to {output_path}")

# Main script
if __name__ == "__main__":
    image_paths = [os.path.join(unseen_images_dir, img) for img in os.listdir(unseen_images_dir) if img.endswith((".png", ".jpg"))]

    for image_path in image_paths:
        print(f"Processing image: {image_path}")

        # Step 1: Predict the mask using the segmentation model
        mask = predict_mask(image_path, segmentation_model, img_size)

        # Step 2: Save the mask
        mask_output_path = os.path.join(output_mask_dir, os.path.basename(image_path).replace(".jpg", "_mask.png"))
        save_mask(mask, mask_output_path)

        # Step 3: Classify the image using the classification model and the mask
        predicted_class, confidence = classify_image_with_mask(image_path, mask, classification_model, categories, img_size)

        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

        # Optional: Visualize the results
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="viridis")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, f"Class: {predicted_class}\nConfidence: {confidence:.2f}%", fontsize=12, ha='center', va='center')
        plt.title("Classification Result")
        plt.axis("off")

        plt.show()
