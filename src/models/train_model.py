from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def train_linear_model(X_train, y_train):
    model = LinearRegression()
    return model.fit(X_train, y_train)

def train_decision_tree(X_train, y_train, depth=3, features=10):
    dt = DecisionTreeRegressor(max_depth=depth, max_features=features, random_state=567)
    return dt.fit(X_train, y_train)

