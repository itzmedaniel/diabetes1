import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("diabetes_dataset_full.csv")

X = df.drop(columns=["Diabetes"])
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "full_diabetes_model_rf.pkl")
print("✅ Model saved as full_diabetes_model_rf.pkl")
