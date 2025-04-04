def prepare_features(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y
