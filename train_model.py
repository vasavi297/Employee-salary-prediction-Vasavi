import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle

# Sample dataset
data = {
    'company': ['Google', 'Microsoft', 'Amazon', 'Google', 'Facebook', 'Apple', 'Netflix'],
    'job': ['Data Scientist', 'Software Engineer', 'Web Developer', 'Data Analyst', 'Machine Learning Engineer', 'Software Engineer', 'Data Scientist'],
    'degree': ['Masters', 'Bachelors', 'Bachelors', 'PhD', 'Masters', 'Masters', 'PhD'],
    'experience': [5, 3, 2, 6, 7, 4, 8],
    'salary': [150000, 120000, 90000, 130000, 160000, 140000, 170000]
}
df = pd.DataFrame(data)

# Define features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Preprocessing
categorical_features = ['company', 'job', 'degree']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
