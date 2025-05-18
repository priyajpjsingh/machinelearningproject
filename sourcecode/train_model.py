import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the CSV files from the data folder
df1 = pd.read_csv("data/coin_gecko_2022-03-16.csv")
df2 = pd.read_csv("data/coin_gecko_2022-03-17.csv")

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Select features and target column
features = ['1h', '24h', '7d', '24h_volume', 'mkt_cap']
target = 'price'

# Drop rows with missing values
df = df.dropna(subset=features + [target])

# Prepare features and target variables
X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

# Save the model and scaler
joblib.dump(model, "model/liquidity_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# Output model performance
score = model.score(X_test, y_test)
print(f"✅ Model trained! R² score on test data: {score:.4f}")
