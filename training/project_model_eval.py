import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import subprocess

# Load and augment data
def prepare_data():
    # Load STL-10
    (train_ds, test_ds), info = tfds.load('stl10', split=['train', 'test'],
                                        as_supervised=True, with_info=True)

    # Convert to numpy arrays and grayscale
    train_images, train_labels = [], []
    for img, label in tfds.as_numpy(train_ds):
        train_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        train_labels.append(label)

    test_images, test_labels = [], []
    for img, label in tfds.as_numpy(test_ds):
        test_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        test_labels.append(label)

    # Normalize and reshape
    train_images = np.array(train_images).reshape(-1, 96, 96, 1) / 255.0
    test_images = np.array(test_images).reshape(-1, 96, 96, 1) / 255.0

    # Binary labels (airplane=1)
    train_labels = (np.array(train_labels) == 0).astype(np.uint8)
    test_labels = (np.array(test_labels) == 0).astype(np.uint8)

    # Augment airplane images (3x)
    airplane_images = train_images[train_labels == 1]
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    augmented = []
    for img in airplane_images:
        for _ in range(3):
            augmented.append(datagen.random_transform(img))

    # Combine datasets
    train_images = np.concatenate([train_images, augmented])
    train_labels = np.concatenate([train_labels, np.ones(len(augmented))])

    return train_images, train_labels, test_images, test_labels

# Create compact model (~50K params)
def build_model():
    model = Sequential([
        Conv2D(4, (3,3), activation='relu', input_shape=(96,96,1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(8, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(16, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model
def preprocess(image, label, image_size=(96, 96)):
    """Resize, convert to grayscale, normalize, and return numpy arrays."""
    image = tf.image.resize(image, image_size)  # Resize using TensorFlow (faster than OpenCV)
    image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
    image = image / 255.0  # Normalize

    label = tf.cast(label == 0, tf.int32)  # Binary label (1 if airplane, else 0)

    return image, label

def load_stl10_data(image_size=(96, 96), batch_size=32):
    # Load dataset
    dataset, info = tfds.load("stl10", split=["train", "test"], as_supervised=True, with_info=True)

    # Apply preprocessing in parallel using .map()
    train_ds = dataset[0].map(lambda img, lbl: preprocess(img, lbl, image_size)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = dataset[1].map(lambda img, lbl: preprocess(img, lbl, image_size)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Convert dataset to NumPy arrays
    train_data = list(tfds.as_numpy(train_ds))
    test_data = list(tfds.as_numpy(test_ds))

    # Unpack images and labels **correctly**
    x_train, y_train = zip(*train_data)
    x_test, y_test = zip(*test_data)

    # Convert to NumPy arrays (using np.vstack for images and np.hstack for labels)
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)  # Fixes the "inhomogeneous shape" issue
    x_test = np.vstack(x_test)
    y_test = np.hstack(y_test)

    return (x_train, y_train), (x_test, y_test)

def evaluate_model(model, image_size=(96, 96), batch_size=32):
    # Load ONLY the test set (using the correct split name)
    test_ds = tfds.load("stl10", split="test", as_supervised=True)

    # Preprocess test set
    test_ds = test_ds.map(lambda img, lbl: preprocess(img, lbl, image_size))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Convert to numpy arrays
    test_data = list(tfds.as_numpy(test_ds))
    x_test, y_test = zip(*test_data)
    x_test = np.vstack(x_test)
    y_test = np.hstack(y_test)

    print(f"\nTest Set Composition:")
    print(f"Airplanes: {np.sum(y_test == 1)} images")
    print(f"Non-airplanes: {np.sum(y_test == 0)} images")

    # Evaluate binary classifier
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Predictions (sigmoid outputs)
    y_pred = model.predict(x_test, verbose=0)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_class)

    # Metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Loss: {test_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Not Airplane", "Airplane"],
                yticklabels=["Not Airplane", "Airplane"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix (Threshold=0.5)')
    plt.show()

# Main
train_images, train_labels, test_images, test_labels = prepare_data()
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1)

model = build_model()
model.summary()

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=12,
          batch_size=32,
          class_weight={0:1, 1:3})

# Save final model
model.save('trained_model.h5')
model = tf.keras.models.load_model('trained_model.h5')
evaluate_model(model)