import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("personality_data.csv")

# Encode categorical data (if necessary)
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# Split the dataset into features and labels
X = df.drop(columns=["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
y = df[["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model using joblib
import joblib
joblib.dump(model, "personality_model.pkl")
