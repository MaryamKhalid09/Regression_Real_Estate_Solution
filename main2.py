# main2.py

import os
from src.data.make_dataset import load_data
from src.features.build_features import prepare_features
from src.data.split import split_data
from src.models.train_model import train_linear_model, train_decision_tree
from src.models.predict_model import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    print("Current working directory:", os.getcwd())

    # Set path to dataset
    file_path = "data/processed/final.csv"

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("Loading data...")
    df = load_data(file_path)
    print("Columns in dataset:", df.columns.tolist())

    print("Preparing features...")
    # Use 'price' as the target column
    X, y = df.drop('price', axis=1), df['price']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Linear Regression model...")
    lr_model = train_linear_model(X_train, y_train)
    evaluate_model(lr_model, X_train, y_train, "Train (Linear)")
    evaluate_model(lr_model, X_test, y_test, "Test (Linear)")

    print("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train)
    evaluate_model(dt_model, X_train, y_train, "Train (Decision Tree)")
    evaluate_model(dt_model, X_test, y_test, "Test (Decision Tree)")

if __name__ == "__main__":
    main()
