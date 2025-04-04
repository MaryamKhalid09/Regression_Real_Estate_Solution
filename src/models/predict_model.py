from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X, y, dataset_name=""):
    predictions = model.predict(X)
    mae = mean_absolute_error(predictions, y)
    print(f"{dataset_name} MAE: {mae}")
    return mae
