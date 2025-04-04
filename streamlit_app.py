import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final.csv")
    return df

# Train and evaluate model
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, predictions, mae, r2

# Streamlit UI
st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("🏠 Real Estate Price Prediction Dashboard")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show column names for verification
st.text("Columns: " + ", ".join(df.columns))

# Assuming 'price' is the target variable
try:
    X = df.drop("price", axis=1)
    y = df["price"]
except KeyError:
    st.error("Column 'price' not found in dataset. Please check your CSV file.")
    st.stop()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_type = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree"])

if model_type == "Linear Regression":
    model = LinearRegression()
else:
    model = DecisionTreeRegressor(max_depth=3, random_state=42)

# Train and evaluate
trained_model, predictions, mae, r2 = train_model(model, X_train, X_test, y_train, y_test)

# Show performance metrics
st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# Line chart: Actual vs Predicted
st.subheader("Actual vs Predicted Prices")
results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})
st.line_chart(results_df.reset_index(drop=True))
