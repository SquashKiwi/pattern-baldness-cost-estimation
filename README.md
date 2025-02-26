# Male Pattern Baldness Classification

_"Hair Today, Gone Tomorrow: A Deep Dive into AI-Driven Baldness Detection"_

## 🚀 Project Overview

Male Pattern Baldness (MPB), or androgenetic alopecia, affects 50% of men by the age of 50 and can begin as early as the teenage years. However, current diagnosis methods rely on manual inspection and subjective assessments, leading to inconsistencies and late detection.

This project introduces an AI-powered solution that automates MPB classification based on the Norwood Scale (a standard measure of the severity of male pattern baldness), eliminating the need for expensive genetic testing and dermatology consultations.

![the norwood scale](assets/norwood_scale.png)

## 🏆 Key Features

✔ AI-based Classification: Uses deep learning to detect MPB stages. <br>
✔ Non-Invasive & Accessible: No need for expensive medical tests. <br>
✔ Scalable & Efficient: Can process large datasets with minimal human intervention. <br>
✔ User Input Refinement: Adjusts results based on age, smoking habits, and race for a more personalized diagnosis. <br>

## ✅ Objectives

🔹 Automates phenotypic analysis of MPB stages using the Norwood Scale. <br>
🔹 Uses deep learning models for classification. <br>
🔹 Provides a reliable, accessible, and cost-effective AI-driven tool. <br>
🔹 Provides the user with a cost estimate of transplant surgeries in various regions. <br>

## 📊 Data Overview & EDA

### 📌 Initial Dataset:

🔹 125 high-resolution scalp assets (~32 per group).

### 📌 Data Augmentation:

-  Expanded to 775 assets using:
- Brightness & saturation adjustments
- Geometric transformations (flipping, rotation)

### 📌 Key Features:

- ✔ MPB Stages: Classified into 4 groups based on the Norwood Scale (as shown in the image above).
- ✔ Additional Attributes: Race, Age, Smoking habits.
- ✔ Preprocessing: Resized to 256×256, converted to grayscale, and normalized.

### 📌 Correlation Analysis:

- ✅ Age & Baldness: Strong positive correlation (r=0.88) – significant predictor of progression.
- ✅ Smoking & Baldness: Moderate correlation (r=0.64) – notable association.
- ✅ Race & Baldness: Negative correlation (r=-0.12) – possibly dataset-dependent.

![correlation analysis](assets/correlation_matrix.png)

## 🧠 Model Overview

We developed three models for Male Pattern Baldness (MPB) classification:

1. Annotation Model
2. FEv1
3. FEv4

Unlike FE v1 and FE v4, which classify directly from assets, the Annotation Model first segments the scalp and hair using U-Net, creating a preprocessed mask. This mask is then classified separately, making the approach more interpretable but requiring an additional annotation step.

A block Diagram of FEv4:
<br>
![FEv4 Block Diagram](assets/FEv4_block.png)

## 🔬 Model Comparison

| Feature/Aspect           | Annotation Model                                                    | FE v1                                                               | FE v4                                                                             |
| ------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Architecture**         | U-Net for segmentation                                              | Custom CNN (11 layers)                                              | Custom CNN (with dense layers)                                                    |
| **Key Layers**           | Encoder-decoder structure, Pixel-wise classification                | 3 Conv layers (32, 64, 128 filters), Flatten, Dense layers, Dropout | 3 Conv layers (32, 64, 128 filters with Batch Norm), Dense (128 neurons), Dropout |
| **Activation Functions** | Softmax for segmentation                                            | ReLU (Conv layers)                                                  | ReLU (Conv layers), Softmax (output layer)                                        |
| **Training Method**      | Adam Optimizer, Loss function not specified                         | Adam Optimizer, Sparse Categorical Crossentropy loss                | Cosine Annealing, Early Stopping, ModelCheckpoint                                 |
| **Output**               | Segmentation masks with pixel-wise classes: background, hair, scalp | Multi-class classification                                          | Multi-class classification (4 classes for baldness stages)                        |
| **Performance Accuracy** | 63%                                                                 | 83%                                                                 | **96%**                                                                           |

<br>

![ROC Curves](assets/roc_curves.png)

## 🧠 Subclassification Layer

The refined Norwood prediction function refines baldness stage detection by integrating:

- 1️⃣ User inputs – Age, Smoking status, Race.
- 2️⃣ Weighted calculations – Normalized values influence prediction.
- 3️⃣ Threshold adjustments – Fine-tunes the model output for personalized stage classification.
