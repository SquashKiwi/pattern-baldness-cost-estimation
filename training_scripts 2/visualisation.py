import os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Parameters
DATA_DIR = "./augmented"  # Dataset directory
MODEL_PATH = "fe4.keras"  # Path to saved model
CATEGORIES = ["group1", "group2", "group3", "group4"]  # Class labels
IMG_SIZE = (256, 256)  # Image size

# Load dataset
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

# Plot ROC Curve
def plot_multiclass_roc(y_test, y_probs, categories):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(categories)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i in range(len(categories)):
        plt.plot(fpr[i], tpr[i], label=f"ROC curve (AUC = {roc_auc[i]:.2f}) for {categories[i]}")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, categories):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X, Y = load_dataset(DATA_DIR)

    # Train-val-test split
    from sklearn.model_selection import train_test_split
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Predict probabilities for the test set
    print("Generating predictions...")
    Y_test_cat = to_categorical(Y_test, num_classes=len(CATEGORIES))
    Y_probs = model.predict(X_test)

    # ROC Curve
    print("Plotting ROC curve...")
    plot_multiclass_roc(Y_test_cat, Y_probs, CATEGORIES)

    # Confusion Matrix
    print("Generating confusion matrix...")
    Y_pred = np.argmax(Y_probs, axis=1)
    plot_confusion_matrix(Y_test, Y_pred, CATEGORIES)
