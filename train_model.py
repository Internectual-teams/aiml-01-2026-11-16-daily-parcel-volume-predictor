import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

HUB_CODES = {
    "HUB001": 0,
    "HUB002": 1,
    "HUB003": 2
}

# Load dataset
data = pd.read_csv("processed_data.csv")

# Extract date features
data["delivery_date"] = pd.to_datetime(data["delivery_date"])
data["day"] = data["delivery_date"].dt.day
data["month"] = data["delivery_date"].dt.month

# Encode delivery hubs
data["hub_code"] = data["hub_id"].map(HUB_CODES)

# Prepare training data
X = data[["day", "month", "hub_code"]]
y = data["parcel_volume"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("Model saved as model.pkl")