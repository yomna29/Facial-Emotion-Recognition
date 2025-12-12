

# Facial Expression Recognition (FER) with ResNet50

This project implements a deep learning pipeline for Facial Expression Recognition (FER) using a fine-tuned **ResNet50** architecture. The workflow involves merging multiple datasets, performing rigorous data balancing (augmentation and undersampling), and training a model using a two-phase transfer learning strategy.

## Project Overview

The notebook constructs a robust emotion classifier by combining the standard **FER2013** dataset with the **Emotion Detection FER** dataset. It addresses class imbalance through targeted augmentation (using Albumentations) and undersampling, followed by training a TensorFlow/Keras model.

**Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Data Pipeline

### 1. Dataset Merging & Splitting
The data is sourced from two Kaggle inputs (`fer2013` and `emotion-detection-fer`) and restructured into a custom directory layout:
* **TRAIN:** Contains **all** images from "Emotion Detection FER" + **75%** of "FER2013" (train set).
* **VAL:** Contains **25%** of "FER2013" (train set).
* **TEST:** Contains the "FER2013" test/private split.

### 2. Data Balancing
To handle significant class imbalances (e.g., "Disgust" having very few samples):
* **Augmentation:** Uses `albumentations` to generate synthetic images (flips, rotation, shear, brightness, contrast, hue shift).
    * *Target:* 1,200 images for "Disgust", 7,000 for all other classes.
    * Augmentation is only applied if the class count is below the target.
* **Undersampling:** Randomly samples classes down to a maximum of 9,000 images to prevent majority class dominance.

## Model Architecture

* **Backbone:** ResNet50 (pre-trained on ImageNet).
* **Custom Head:**
    * Global Average Pooling.
    * Dense Layer (32 units, ReLU activation).
    * Batch Normalization.
    * Dropout (0.4).
    * Output Layer (Softmax, 7 classes).

##  Training Strategy

The model is trained in **two phases** to ensure stability and high performance:

### Phase 1: Head Training
* **Status:** Backbone (ResNet50) is **frozen**.
* **Goal:** Train the custom classification head to adapt to the features.
* **Epochs:** 10.
* **Optimizer:** Adam (LR: 5e-4).

### Phase 2: Fine-Tuning
* **Status:** Top 50 layers of the backbone are **unfrozen** (BatchNormalization layers remain frozen).
* **Goal:** Fine-tune high-level feature extraction specific to facial expressions.
* **Epochs:** 20 (Total 30).
* **Optimizer:** AdamW (or Adam) with Weight Decay and a lower Learning Rate (approx 8e-5).
* **Callbacks:**
    * `EarlyStopping`: Monitors validation loss (patience: 6).
    * `ReduceLROnPlateau`: Halves LR if validation loss stagnates.
    * `IncreaseWeightDecayOnLRDrop`: Custom callback to adjust regularization dynamically.

##  Evaluation Metrics

The model is evaluated using the following metrics:
* **Loss:** Sparse Categorical Crossentropy.
* **Accuracy:** Sparse Categorical Accuracy.
* **Macro Metrics:** Precision, Recall, and F1-Score (averaged across classes).

**Visualization:**
The notebook generates:
1.  **Confusion Matrices:** Seaborn heatmaps for Validation and Test sets.
2.  **Classification Report:** Precision, Recall, and F1-score breakdown per class.
3.  **Distribution Plots:** Bar charts showing dataset distribution before and after augmentation.

##  Dependencies

* Python 3.10+
* TensorFlow / Keras
* Albumentations
* OpenCV
* NumPy, Pandas, Matplotlib, Seaborn
* Shutil (for file manipulation)

## Usage

1.  **Data Setup:** Ensure Kaggle inputs `fer2013` and `emotion-detection-fer` are attached to the environment.
2.  **Run Pipeline:** Execute cells sequentially. The notebook handles directory creation, data copying, and training automatically.
3.  **Outputs:**
    * Processed data is stored in `/kaggle/content/fer2013_by_usage_1`.
    * Confusion matrix images are saved as `confusion_matrix_val.png` and `confusion_matrix_test.png`.

