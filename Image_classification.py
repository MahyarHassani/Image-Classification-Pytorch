import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Unzip the images
!unzip -o seg_test.zip

# Paths and labels
_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Categories
IMGSIZE = (128, 128)  # Resize all images to 128x128 pixels
CNAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Duplicate of _labels for label indexing
X_tr, y_tr = [], []  # Empty lists to store images and labels

def saveimages_inlist():
    """
    Load images from subdirectories corresponding to different categories, resize them to a uniform size (128x128), 
    and return the images along with their labels.

    Returns:
    --------
    X_tr : list
        A list containing all the resized images as numpy arrays.
    y_tr : list
        A list containing the corresponding label indices of the images.
    """
    for label in _labels:
        path = os.path.join('seg_test', label)  # Construct the path for each category
        for img_file in os.listdir(path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                resized_img = cv2.resize(img, IMGSIZE)
                X_tr.append(resized_img)
                label_index = CNAMES.index(label)
                y_tr.append(label_index)

    return X_tr, y_tr  # Return the images and their corresponding label indices

X_tr, y_tr = saveimages_inlist()

# Check the number of color channels
if X_tr and len(X_tr[0].shape) == 3:
    print("Number of channels=", X_tr[0].shape[2])
else:
    raise ValueError("Images are not loaded properly or do not have three channels.")

# Display a few images
plt.figure(figsize=(10, 10))
for i in range(min(9, len(X_tr))):
    plt.subplot(3, 3, i + 1)
    plt.imshow(cv2.cvtColor(X_tr[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title(CNAMES[y_tr[i]])
    plt.axis('off')
plt.show()

def convert_toNumpyarray(X_tr, y_tr):
    """
    Convert the lists of images and labels into numpy arrays and normalize the image data.

    Parameters:
    -----------
    X_tr : list
        A list containing image data, where each image is represented as a 3D array (height, width, channels).
    y_tr : list
        A list containing the corresponding label indices for each image.

    Returns:
    --------
    X_tr : numpy.ndarray
        A numpy array containing all the image data, normalized to a range of [0, 1].
    y_tr : numpy.ndarray
        A numpy array containing the label indices.
    """
    X_tr = np.array(X_tr, dtype='float32') / 255.0  # Convert to numpy array and normalize
    y_tr = np.array(y_tr, dtype='int32')  # Convert to numpy array

    return X_tr, y_tr

X_tr, y_tr = convert_toNumpyarray(X_tr, y_tr)

print("X_tr.shape=", X_tr.shape)
print("y_tr.shape=", y_tr.shape)

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for classification tasks.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer for class predictions.
    """

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(CNAMES))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_fully_connected_network(X_tr, y_tr):
    """
    Train a fully connected neural network on the given training data and evaluate it on a validation set.

    Parameters:
    X_tr (numpy.ndarray): Training data features (input).
    y_tr (numpy.ndarray): Training data labels (output).

    Returns:
    float: The validation accuracy after training.
    """
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        indices = torch.randperm(X_train_tensor.size(0))
        X_train_tensor = X_train_tensor[indices]
        y_train_tensor = y_train_tensor[indices]

        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            del batch_X, batch_y, outputs, loss
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(y_val, val_preds.cpu().numpy())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')

    return val_accuracy

val_accuracy = test_fully_connected_network(X_tr, y_tr)
print("Validation accuracy=", val_accuracy)
