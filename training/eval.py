#eval
import tensorflow as tf
from train import load_data

def evaluate_model():
    # Load the trained model
    model = tf.keras.models.load_model('models/trained_model.h5')

    # Load training and testing data
    train_ds = load_data('dataset')  

    # Print model summary (to show parameters)
    print("Model Summary:")
    model.summary()

    # Evaluate the model on training and testing data
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)

    # Print results
    print("\nNote that due to a reduced dataset, the data is not split into training and test sets,\nas it would be for a larger dataset, and for the project.\nUtilizing a pretrained model or pre-existing dataset would result in a higher model accuracy and lower loss.")
    print("\n\nModel Evaluation:")
    print(f"Training and Testing Accuracy: {train_acc}")
    print(f"Training and Testing Loss: {train_loss}")

if __name__ == "__main__":
    evaluate_model()