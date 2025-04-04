import pandas as pd

def load_data(data_path):
    # Load data from a CSV file
    df = pd.read_csv(data_path)
    return df

