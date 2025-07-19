import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def test_model(model_path='model_train.pkl'):
    # Load the trained model
    model = joblib.load(model_path)

    # Load the digits dataset
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print("Model accuracy:", acc)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    test_model()
