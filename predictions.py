import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("retail_sales_dataset.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values")
print(df.isnull().sum())

# Remove missing values if any
df = df.dropna()

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# Convert Date Column
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# -----------------------------
# Encode Categorical Columns
# -----------------------------
le_gender = LabelEncoder()
le_category = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])

df["Product Category"] = le_category.fit_transform(df["Product Category"])

# -----------------------------
# Features and Target
# -----------------------------
X = df[[
    "Age",
    "Gender",
    "Product Category",
    "Price per Unit",
    "Year",
    "Month",
    "Day"
]]

y = df["Quantity"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score :", r2)

# -----------------------------
# Actual vs Predicted Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.title("Actual vs Predicted Quantity")
plt.grid(True)
plt.show()

# -----------------------------
# Residual Plot
# -----------------------------
errors = y_test - predictions

plt.figure(figsize=(8,6))
plt.scatter(predictions, errors)
plt.axhline(y=0)
plt.xlabel("Predicted Quantity")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# -----------------------------
# Save Predictions
# -----------------------------
results = pd.DataFrame({
    "Actual Quantity": y_test,
    "Predicted Quantity": predictions
})

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved as predictions.csv")

# -----------------------------
# Regression Equation
# -----------------------------
print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coef)

# -----------------------------
# Predict New Customer Example
# -----------------------------
new_customer = pd.DataFrame({
    "Age":[30],
    "Gender":[1],              # Male
    "Product Category":[0],    # Encoded category
    "Price per Unit":[500],
    "Year":[2023],
    "Month":[11],
    "Day":[10]
})

prediction = model.predict(new_customer)

print("\nPredicted Quantity")
print(prediction)

print("\nProject Completed Successfully")
