import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

import cv2
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# 1. Load data
data_directory = "./augmented"  # Update this if your directory differs
categories = ["group1", "group2", "group3", "group4"]
img_size = (256, 256)

def load_data(data_directory, categories, img_size):
    images, labels = [], []
    label_map = {category: idx for idx, category in enumerate(categories)}
    for filename in os.listdir(data_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            group = filename.split("_")[0]
            if group in label_map:
                filepath = os.path.join(data_directory, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size) / 255.0
                images.append(img)
                labels.append(label_map[group])
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels)
    return images, labels

X, y = load_data(data_directory, categories, img_size)
print(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Load the pre-trained model
model_path = "fe4.keras"  # Path to the saved model
model = load_model(model_path)
print("Model loaded successfully.")

# 4. Get predictions on X_test
y_probs = model.predict(X_test)                 # shape: (n_samples, n_classes)
y_pred = np.argmax(y_probs, axis=1)             # predicted class indices

# 5. Plot ROC Curve (one-vs-rest) for each class
y_test_cat = to_categorical(y_test, num_classes=len(categories))

def plot_multiclass_roc(y_true, y_score, class_names, figsize=(8, 6)):
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(class_names)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=figsize)
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i], 
            label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Multiclass)")
    plt.legend(loc="lower right")
    plt.show()

plot_multiclass_roc(y_test_cat, y_probs, categories)

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 7. Classification Report (Precision, Recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))
