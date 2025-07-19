import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def run_inference(model_path='model_train.pkl', n_samples=4):
    # Load trained model
    model = joblib.load(model_path)

    # Load digits dataset and split
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    # Pick N random samples from the test set
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    sample_data = X_test[idx]
    actual = y_test[idx]

    # Predict
    predicted = model.predict(sample_data)

    print("Predicted:", list(predicted))
    print("Actual:", list(actual))

if __name__ == "__main__":
    run_inference()
