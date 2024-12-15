# Image-Classification-Pytorch
Image classification using Pytorch and matplotlib.pyplot
Source: https://www.coursera.org/specializations/applied-machine-learning
# angeImage Classification Pipeline

This repository contains an image classification pipeline built using Python. The pipeline processes images from a dataset, trains a simple neural network for classification, and evaluates the model's performance on a validation set. The main features include data preprocessing, visualization, model training, and accuracy evaluation.

## Features

- **Data Loading and Preprocessing**:

  - Images are loaded from subdirectories corresponding to different categories.
  - Each image is resized to a uniform size of 128x128 pixels.
  - Image data is normalized to improve training performance.

- **Visualization**:

  - Displays a grid of example images from the dataset with their corresponding class labels.

- **Model Architecture**:

  - A simple feedforward neural network with three fully connected layers is used for classification.
  - Supports multiple output classes defined by the dataset categories.

- **Training and Evaluation**:

  - The dataset is split into training and validation sets.
  - The model is trained using the Adam optimizer and cross-entropy loss function.
  - Batch training is implemented with GPU support for faster computations (if available).
  - Validation accuracy is reported after each epoch.

## Requirements

To run this code, you need the following dependencies:

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- PyTorch
- Scikit-learn

Install the required libraries using pip:

```bash
pip install opencv-python-headless numpy matplotlib torch scikit-learn
```

## Usage

1. **Dataset Preparation**:

   - Place your dataset inside a folder named `seg_test`.
   - Each category should be stored in its subdirectory (e.g., `seg_test/buildings`, `seg_test/forest`, etc.).
   - Ensure the dataset is in `.jpg` format.

2. **Run the Code**:

   - Execute the script in your Python environment.

```bash
python image_classification_pipeline.py
```

3. **Outputs**:
   - Displays example images with their class labels.
   - Reports validation accuracy after training the model.

## Model Details

The feedforward neural network consists of:

1. **Input Layer**: Flattens the image data (3 color channels, 128x128 pixels).
2. **Hidden Layers**:
   - Layer 1: 512 neurons with ReLU activation.
   - Layer 2: 256 neurons with ReLU activation.
3. **Output Layer**: Predicts class probabilities for 10 categories (modifiable based on dataset).

## Potential Improvements

- Add support for convolutional neural networks (CNNs) for better accuracy.
- Implement early stopping and learning rate scheduling during training.
- Allow for dataset augmentation to improve model generalization.

## License

This project is open-source and can be modified or redistributed under the terms of the MIT License.


