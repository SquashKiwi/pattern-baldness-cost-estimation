import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Set directories
data_directory = os.path.dirname(os.path.abspath(__file__))  # Current script directory
categories = ["group1", "group2", "group3", "group4"]  # Labels

# Parameters
img_size = (256, 256)  
batch_size = 32
epochs = 20

# Data preparation
def load_data(data_directory, categories, img_size):
    X = []  # Images
    y = []  # Labels

    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_directory, category, "resized")  # Target 'resized' subfolder
        if not os.path.exists(folder_path):
            print(f"Resized folder not found for category: {category}")
            continue

        for img_file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                X.append(img_array)
                y.append(idx)  # Use the index as the label
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")

    X = np.array(X, dtype='float32') / 255.0  # Normalize pixel values to [0, 1]
    y = np.array(y)
    return X, y



# Load the data
print("Loading data...")
X, y = load_data(data_directory, categories, img_size)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Data augmentation
data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
data_gen.fit(X_train)

# Model definition
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(categories), activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the model
model = create_model()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model_path = os.path.join(data_directory, "baldness_classifier.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Prediction function
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    return predicted_class



# Example usage
# test_image = "path_to_test_image.png"
# print(f"Predicted class: {predict_image(test_image)}")
