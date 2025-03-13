#train
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

dataset_dir = './dataset'

def build_model(input_shape=(320, 320, 3)):
    model = models.Sequential([
        # Input preprocessing
        layers.Rescaling(1./255, input_shape=input_shape),
        # Data augmentation to attempt to achieve a better model
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),

        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SeparableConv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Dropout regularization
        layers.Dense(1, activation='sigmoid')
    ])
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_data(dataset_dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=(320, 320),
        batch_size=2)
    return dataset

if __name__ == "__main__":
    # Load data
    train_ds = load_data('dataset')

    # Build and train model
    model = build_model()
    model.fit(train_ds, epochs=12)

    # Save model
    model.save('models/trained_model.h5')